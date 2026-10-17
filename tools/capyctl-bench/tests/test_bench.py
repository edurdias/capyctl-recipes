"""Unit tests for capyctl-bench. Run from tools/capyctl-bench:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import capyctl_bench as cb  # noqa: E402
import fake_server  # noqa: E402

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def run_cli(*argv) -> int:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        return cb.main([str(a) for a in argv])


class StatsTest(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(cb.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(cb.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(cb.percentile([10, 20], 95), 19.5)
        self.assertIsNone(cb.percentile([None], 50))

    def test_spread_ignores_none(self):
        self.assertEqual(cb.spread([3, None, 1, 2]), {"median": 2, "min": 1, "max": 3, "n": 3})
        self.assertEqual(cb.spread([])["n"], 0)

    def test_draft_counts(self):
        self.assertEqual(cb.find_draft_counts({"tensorfold": {"drafted": 10, "accepted": 7}}), (10, 7))
        self.assertEqual(cb.find_draft_counts({"num_draft_tokens": 8, "num_accepted_tokens": 4,
                                               "acceptance_rate": 0.5}), (8, 4))
        self.assertIsNone(cb.find_draft_counts({"tensorfold": {"token_sha": "ab"}}))
        reqs = [{"extras": {"x": {"drafted": 10, "accepted": 5}}},
                {"extras": {"x": {"drafted": 30, "accepted": 25}}}]
        self.assertAlmostEqual(cb.acceptance(reqs), 0.75)

    def test_sizes_and_levels(self):
        self.assertEqual(cb.parse_sweep("0.5k,1k,128k,300"), [500, 1000, 128000, 300])
        self.assertEqual(cb.size_label(500), "0.5k")
        self.assertEqual(cb.size_label(128000), "128k")
        self.assertEqual(cb.parse_levels("1-4"), [1, 2, 3, 4])
        self.assertEqual(cb.parse_levels("1,2,8"), [1, 2, 8])

    def test_filler_is_deterministic(self):
        self.assertEqual(cb.filler_text(5, 42), cb.filler_text(5, 42))
        self.assertNotEqual(cb.filler_text(5, 42), cb.filler_text(5, 43))

    def test_api_key_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "credentials"
            p.write_text("principal: me\napi_key: sk-abc123\n")
            self.assertEqual(cb.read_api_key(str(p)), "sk-abc123")
            p.write_text(json.dumps({"api_key": "sk-json"}))
            self.assertEqual(cb.read_api_key(str(p)), "sk-json")

    def test_endpoint_redacted(self):
        ep = cb.Endpoint("http://gpu-box.local:8443/v1")
        self.assertEqual(ep.redacted(), "http://<host>:8443/v1")


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.srv = fake_server.start(fake_server.FakeConfig(
            token_delay=0.005, chunk_tokens=2, reasoning_tokens=4, api_key="k"))

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def test_one_request(self):
        ep = cb.Endpoint(self.srv.url, "k")
        rec = cb.stream_chat(ep, {"model": "sample-model", "messages": [{"role": "user", "content": "a b c d"}],
                                  "max_tokens": 20, "stream": True,
                                  "stream_options": {"include_usage": True}}, record_chunks=True)
        self.assertIsNone(rec["error"])
        self.assertEqual(rec["prompt_tokens"], 8)
        self.assertEqual(rec["completion_tokens"], 20)
        self.assertEqual(rec["token_chunks"], 10)
        self.assertEqual(len(rec["chunk_times_s"]), 10)
        # reasoning sent as both reasoning_content and reasoning counts once
        self.assertEqual(rec["output_chars"], len("".join(f"w{i} " for i in range(20))))
        # 9 chunk gaps of ~10 ms carry 19 tokens: about 190 tok/s, 5 ms per token
        self.assertGreater(rec["decode_tps"], 100)
        self.assertLess(rec["decode_tps"], 260)
        self.assertAlmostEqual(rec["tpot_ms"], 1000 / rec["decode_tps"])
        self.assertEqual(rec["extras"]["sample_engine"]["drafted"], 40)
        self.assertTrue(rec["extras"]["capyctl_bench_sample"])
        self.assertLessEqual(rec["ttft_s"], rec["total_s"])

    def test_bad_key_is_an_error(self):
        rec = cb.stream_chat(cb.Endpoint(self.srv.url, "wrong"),
                             {"model": "sample-model", "messages": [], "max_tokens": 4, "stream": True})
        self.assertTrue(rec["error"].startswith("HTTP 401"))


class CutStreamTest(unittest.TestCase):
    """A stream that ends without [DONE] (a proxy closed it) is an error, with or without tokens."""

    def cut(self, after: int) -> dict:
        srv = fake_server.start(fake_server.FakeConfig(cut_after=after))
        try:
            return cb.stream_chat(cb.Endpoint(srv.url), {"model": "sample-model", "max_tokens": 8, "stream": True,
                                                         "messages": [{"role": "user", "content": "a"}]})
        finally:
            srv.shutdown()
            srv.server_close()

    def test_cut_before_any_token(self):
        rec = self.cut(0)
        self.assertIn("stream ended without [DONE]", rec["error"])
        self.assertIn("no content or reasoning tokens", rec["error"])
        self.assertIn("1 chunk", rec["error"])

    def test_cut_after_tokens(self):
        rec = self.cut(3)
        self.assertIsNotNone(rec["error"])
        self.assertIn("stream ended without [DONE]", rec["error"])
        self.assertIn("3 token chunks", rec["error"])
        self.assertEqual(rec["token_chunks"], 3)


class HiddenFirstTokenTest(unittest.TestCase):
    """Calibration needs only the prompt token count: its one-token reply may carry no text."""

    def test_context_sweep_calibrates(self):
        srv = fake_server.start(fake_server.FakeConfig(hidden_tokens=1, api_key="secret"))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                key = Path(tmp) / "credentials"
                key.write_text("api_key: secret\n")
                out = Path(tmp) / "hidden.json"
                rc = run_cli("run", "--endpoint", srv.url, "--api-key-file", key, "--model", "sample-model",
                             "--label", "Engine C (sample)", "--context-sweep", "0.5k", "--runs", "1", "--max-tokens", "8",
                             "--concurrency", "1", "--rounds", "1", "--concurrency-max-tokens", "8",
                             "--out", out)
                self.assertEqual(rc, 0)
                data = json.loads(out.read_text())
                self.assertGreater(data["context_sweep"]["calibration"]["per_sentence_tokens"], 0)
        finally:
            srv.shutdown()
            srv.server_close()



class RunAndReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.results = []
        for label, delay in (("Engine A (sample)", 0.002), ("Engine B (sample)", 0.003)):
            srv = fake_server.start(fake_server.FakeConfig(token_delay=delay, api_key="secret"))
            try:
                mem = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
                    "import urllib.request;print(urllib.request.urlopen("
                    f"'http://127.0.0.1:{srv.server_address[1]}/debug/memory').read().decode())")
                out = cls.dir / f"engine-{label.split()[1].lower()}.json"
                key = cls.dir / "credentials"
                key.write_text("api_key: secret\n")
                rc = run_cli("run", "--endpoint", srv.url, "--api-key-file", key, "--model", "sample-model",
                             "--label", label, "--context-sweep", "0.5k,1k,2k", "--runs", "2",
                             "--max-tokens", "16", "--concurrency", "1-3", "--rounds", "2",
                             "--concurrency-max-tokens", "24", "--memory-cmd", mem,
                             "--memory-interval", "0.05", "--meta", "capyctl=0.0.0-test",
                             "--meta", "engine=fake", "--out", out)
                assert rc == 0
                cls.results.append(out)
                cls.requests = len(srv.requests)
            finally:
                srv.shutdown()
                srv.server_close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_results_schema(self):
        data = json.loads(self.results[0].read_text())
        self.assertEqual(data["schema_version"], cb.SCHEMA_VERSION)
        self.assertEqual(data["mode"], "chat")
        self.assertEqual(data["label"], "Engine A (sample)")
        self.assertEqual(data["endpoint"].split(":")[0], "http")
        self.assertIn("<host>", data["endpoint"])
        self.assertNotIn("127.0.0.1", json.dumps(data["endpoint"]))
        self.assertEqual(data["meta"], {"capyctl": "0.0.0-test", "engine": "fake", "sample": True})
        pts = data["context_sweep"]["points"]
        self.assertEqual([p["label"] for p in pts], ["0.5k", "1k", "2k"])
        for p in pts:
            self.assertEqual(len(p["requests"]), 2)
            self.assertEqual(p["summary"]["errors"], 0)
            actual = p["summary"]["prompt_tokens"]["median"]
            # calibrated sizing lands within 10% of the target
            self.assertLess(abs(actual - p["target_prompt_tokens"]) / p["target_prompt_tokens"], 0.10)
            self.assertIsNotNone(p["summary"]["memory_peak_gib"]["median"])
            self.assertAlmostEqual(p["summary"]["draft_acceptance"], 0.6, places=1)
        # every run has its own seed: no two prompts repeat (no prefix-cache reuse)
        self.assertGreater(pts[1]["summary"]["prompt_tokens"]["median"], pts[0]["summary"]["prompt_tokens"]["median"])
        levels = data["concurrency"]["levels"]
        self.assertEqual([lv["concurrency"] for lv in levels], [1, 2, 3])
        for lv in levels:
            self.assertEqual(len(lv["rounds"]), 2)
            self.assertEqual(len(lv["rounds"][0]["requests"]), lv["concurrency"])
            self.assertIsNotNone(lv["summary"]["aggregate_tps"]["median"])
            self.assertIsNotNone(lv["summary"]["ttft_p95_s"])
        # aggregate throughput grows with streams on the fake server
        agg = [lv["summary"]["aggregate_tps"]["median"] for lv in levels]
        self.assertGreater(agg[2], agg[0])
        # levels continue through the prompt set instead of restarting at its first prompt,
        # so a later level does not re-send what an earlier one left in the prefix cache
        firsts = [lv["rounds"][0]["requests"][0]["prompt_id"] for lv in levels]
        self.assertEqual(len(set(firsts)), len(firsts), firsts)
        self.assertTrue(data["memory"]["samples"])

    def test_report(self):
        out = self.dir / "report"
        rc = run_cli("report", *self.results, "--out", out, "--title", "Fake A vs B")
        self.assertEqual(rc, 0)
        page = (out / "report.html").read_text()
        self.assertIn("<table>", page)
        self.assertIn("Engine A (sample)", page)
        self.assertIn("Engine B (sample)", page)
        self.assertIn("Measured through CapyCTL", page)
        self.assertIn("data:image/webp;base64,", page)
        self.assertGreaterEqual(page.count("<svg"), 9)
        self.assertIn("Draft acceptance", page)
        self.assertIn("@media print", page)
        summary = (out / "summary.md").read_text()
        self.assertIn("| Context | Engine A (sample) | Engine B (sample) |", summary)
        self.assertRegex(summary, r"\([+-]\d+\.\d%\)")
        rows = (out / "data.csv").read_text().splitlines()
        self.assertEqual(rows[0], "series,section,point,x,metric,unit,median,min,max,n")
        self.assertGreater(len(rows), 50)
        self.assertTrue((out / "report.zip").exists())
        pngs = sorted((out / "charts").glob("*.png")) if (out / "charts").exists() else []
        if HAVE_MPL:
            self.assertGreaterEqual(len(pngs), 9)
        else:
            self.assertEqual(pngs, [])

    def assert_themed_page(self, page: str):
        # theme switcher: a toggle button, the choice kept in localStorage behind try/catch
        self.assertIn('class="theme"', page)
        self.assertIn("capyToggleTheme()", page)
        self.assertIn("localStorage.getItem", page)
        self.assertIn("try{localStorage.setItem", page)
        # light on first visit whatever the OS prefers; dark only through data-theme
        self.assertIn('<html lang="en" data-theme="light">', page)
        self.assertIn("var t='light'", page)
        self.assertNotIn("prefers-color-scheme", page)
        self.assertIn(':root{--bg:#fafaf9;', page)
        self.assertIn(':root[data-theme="dark"]{--bg:#1c1917;', page)
        # print is always light
        self.assertRegex(page, r'@media print\{\s*:root,:root\[data-theme="dark"\]\{--bg:#fafaf9;')
        # both logos embedded, one per theme
        self.assertIn('class="logo-light" src="data:image/webp;base64,', page)
        self.assertIn('class="logo-dark" src="data:image/webp;base64,', page)

    def test_sample_banner(self):
        out = self.dir / "sample"
        run_cli("report", *self.results, "--out", out, "--summary", "--no-png", "--no-zip")
        for page in ((out / "report.html").read_text(), (out / "summary" / "summary.html").read_text()):
            self.assertIn('<p class="sample" role="note">SAMPLE DATA — not a measurement</p>', page)
        self.assertIn("SAMPLE DATA", (out / "summary.md").read_text())
        # a results file without meta.sample renders no banner
        data = json.loads(self.results[0].read_text())
        data["meta"].pop("sample")
        real = self.dir / "not-sample.json"
        real.write_text(json.dumps(data))
        out2 = self.dir / "not-sample"
        run_cli("report", real, "--out", out2, "--summary", "--no-png", "--no-zip")
        for page in ((out2 / "report.html").read_text(), (out2 / "summary" / "summary.html").read_text()):
            self.assertNotIn('class="sample"', page)
        self.assertNotIn("SAMPLE DATA", (out2 / "summary.md").read_text())

    def test_meta_sample_flag(self):
        self.assertEqual(cb.parse_meta(["sample=true", "gpu=x"]), {"sample": True, "gpu": "x"})

    def test_report_theme(self):
        out = self.dir / "themed"
        run_cli("report", *self.results, "--out", out, "--no-zip")
        self.assert_themed_page((out / "report.html").read_text())

    def test_summary(self):
        out = self.dir / "shared"
        rc = run_cli("report", *self.results, "--out", out, "--summary", "--title", "Fake A vs B")
        self.assertEqual(rc, 0)
        page = (out / "summary" / "summary.html").read_text()
        self.assert_themed_page(page)
        self.assertIn("Fake A vs B", page)
        self.assertIn("measured through CapyCTL", page)
        self.assertIn("github.com/edurdias/capyctl-recipes", page)
        self.assertIn("Generation, real prompts", page)
        self.assertIn("Generation by prompt length", page)
        self.assertGreaterEqual(page.count('class="card"'), 3)
        # N series: the second series' bars carry the change against the first
        self.assertRegex(page, r'class="barpct"[^>]*>[+-]\d+\.\d%<')
        self.assertIn('href="summary/summary.html"', (out / "report.html").read_text())
        names = sorted(p.name for p in (out / "summary").glob("*.png"))
        if HAVE_MPL:
            self.assertEqual(names, ["summary-tall-dark.png", "summary-tall.png",
                                     "summary-wide-dark.png", "summary-wide.png"])
            import struct
            head = (out / "summary" / "summary-wide.png").read_bytes()[16:24]
            self.assertEqual(struct.unpack(">II", head), (2400, 1350))
            head = (out / "summary" / "summary-tall.png").read_bytes()[16:24]
            self.assertEqual(struct.unpack(">II", head), (2160, 2700))
        else:
            self.assertEqual(names, [])

    def test_summary_one_series_has_no_percentages(self):
        out = self.dir / "one"
        run_cli("report", self.results[0], "--out", out, "--summary", "--no-png")
        page = (out / "summary" / "summary.html").read_text()
        self.assertIn("Engine A (sample)", page)
        self.assertNotIn('class="barpct"', page)
        self.assertIn('class="barval"', page)

    def test_one_stream_count(self):
        # A concurrency section with a single stream count (for example C1 only, from an engine
        # that serves one request at a time) has no "against streams" line to draw.
        data = json.loads(self.results[0].read_text())
        data["concurrency"]["levels"] = data["concurrency"]["levels"][:1]
        data["concurrency"]["settings"]["levels"] = [1]
        one = self.dir / "one-level.json"
        one.write_text(json.dumps(data))
        out = self.dir / "one-level"
        run_cli("report", one, "--out", out, "--summary", "--no-png", "--no-zip")
        page = (out / "summary" / "summary.html").read_text()
        self.assertNotIn("Throughput vs streams", page)
        self.assertNotIn("1–1 streams", page)
        self.assertIn("1 stream, max_tokens 24", page)
        self.assertIn("Prompt processing", page)
        # report.html: no one-point charts against streams; the C1 numbers stay in the tables
        report = (out / "report.html").read_text()
        self.assertNotIn('<h2 id="concurrency">', report)
        self.assertIn("Aggregate", report)
        series = cb.load_series([one])
        self.assertFalse([sp for sp in cb.chart_specs(series) if sp["section"] == "concurrency"])

    def test_footnote_uses_point_labels(self):
        # An adapted run with 2^n targets (512 ... 131072) labels its points 0.5k ... 128k;
        # the footnote names those labels, not 0.512k-131.072k.
        data = json.loads(self.results[0].read_text())
        for p, (t, lab) in zip(data["context_sweep"]["points"], ((512, "0.5k"), (1024, "1k"), (2048, "2k"))):
            p["target_prompt_tokens"], p["label"] = t, lab
        data["context_sweep"]["settings"]["targets"] = [512, 1024, 2048]
        pow2 = self.dir / "pow2.json"
        pow2.write_text(json.dumps(data))
        out = self.dir / "pow2"
        run_cli("report", pow2, "--out", out, "--summary", "--no-png", "--no-zip")
        page = (out / "summary" / "summary.html").read_text()
        self.assertIn("prompts 0.5k–2k tokens", page)

    def test_headline_uses_concurrency_when_present(self):
        # Real prompts at the lowest and highest stream count lead the summary; the context
        # sweep's filler prompts (easy to draft) stay in the line charts.
        series = cb.load_series(self.results)
        title, groups = cb.headline_groups(series)
        self.assertEqual(title, "Generation, real prompts")
        self.assertEqual([g["label"] for g in groups], ["1 stream", "3 streams"])
        for g, c in zip(groups, (1, 3)):
            for s, v in g["bars"]:
                want = next(cb.cc_value(lv, "aggregate_tps")["median"] for lv in s.levels if lv["concurrency"] == c)
                self.assertEqual(v, want)
        panels = cb.summary_panels(series)
        self.assertEqual(panels[0]["kind"], "bars")
        self.assertIn("Generation by prompt length", [p["title"] for p in panels])
        self.assertLessEqual(len(panels), 4)

    def test_headline_falls_back_to_context_sweep(self):
        data = json.loads(self.results[0].read_text())
        data.pop("concurrency")
        ctx = self.dir / "ctx-only.json"
        ctx.write_text(json.dumps(data))
        series = cb.load_series([ctx])
        title, groups = cb.headline_groups(series)
        self.assertEqual(title, "Generation, one stream")
        self.assertEqual([g["label"] for g in groups], ["1k prompt", "2k prompt"])
        self.assertNotIn("Generation by prompt length", [p["title"] for p in cb.summary_panels(series)])

    def test_bar_labels_fit_their_slot(self):
        # Three series with long change labels: each label is shrunk to its bar's slot, so
        # neighbours never run into each other ("+23.8%+20.9%").
        texts = ["+123.8%", "-20.9%", "+1,234"]
        for slot in (35.0, 60.0, 90.0):
            size = cb.fit_font(texts, 20, slot)
            for t in texts:
                self.assertLessEqual(cb.text_width_em(t) * size, slot + 1e-9)
        # room to spare: the requested size is kept
        self.assertEqual(cb.fit_font(["56.9"], 18, 500), 18)
        self.assertGreater(cb.text_width_em("+23.8%"), cb.text_width_em("23.8"))

    def test_summary_three_series(self):
        third = json.loads(self.results[1].read_text())
        third["label"] = "Engine C (sample)"
        for lv in third["concurrency"]["levels"]:
            lv["summary"]["aggregate_tps"]["median"] *= 1.2345
        p = self.dir / "engine-c.json"
        p.write_text(json.dumps(third))
        out = self.dir / "three"
        rc = run_cli("report", *self.results, p, "--out", out, "--summary", "--no-zip")
        self.assertEqual(rc, 0)
        page = (out / "summary" / "summary.html").read_text()
        self.assertEqual(page.count('class="barpct"'), 4)
        # SVG labels: the change sits under the value, both within the bar's slot
        self.assertRegex(page, r'class="barpct"[^>]*style="font-size:[\d.]+px"')

    def test_bar_headroom(self):
        # Bar values carry a change line above them with several series; the tallest bar's
        # labels must stay under the panel's unit caption, however short the panel is.
        # labels 40 pt high in a 100 pt panel: the bar may fill at most 60% of it
        self.assertGreaterEqual(cb.bar_top(100.0, 40, 100), 100 / 0.6)
        # a tall panel needs little room; a panel shorter than its labels still leaves some bar
        self.assertLess(cb.bar_top(100.0, 40, 1000), 115)
        self.assertLessEqual(cb.bar_top(100.0, 400, 100), 100 / 0.2 * 1.05)

    def test_line_chart_headroom(self):
        # The highest point sits below the top of a PNG chart, also when every point has one value.
        self.assertGreater(cb.y_top([142.2]), 142.2)
        self.assertGreater(cb.y_top([3.0, 140.0]), 140.0)
        self.assertEqual(cb.y_top([]), 1)

    def test_version_1_results_still_render(self):
        # A chat results file from before `mode` existed (schema 1) reads as a chat run.
        data = json.loads(self.results[0].read_text())
        data["schema_version"] = 1
        data.pop("mode")
        old = self.dir / "v1.json"
        old.write_text(json.dumps(data))
        out = self.dir / "v1"
        self.assertEqual(run_cli("report", old, "--out", out, "--summary", "--no-png", "--no-zip"), 0)
        self.assertIn("Context sweep", (out / "report.html").read_text())
        data["schema_version"] = 99
        old.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            cb.load_series([old])

    @unittest.skipIf(HAVE_MPL, "matplotlib is installed")
    def test_report_without_matplotlib_says_so(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cb.main(["report", str(self.results[0]), "--out", str(self.dir / "nompl")])
        self.assertIn("matplotlib is not installed", err.getvalue())
        self.assertIn("uv run --with matplotlib", err.getvalue())


def fake_pins():
    """ACCURACY_SETS pinned to the fake server's copies of the sets."""
    return [mock.patch.dict(cb.ACCURACY_SETS[name], {"sha256": cb.rows_hash(fake_server.fake_rows(repo)[1])})
            for name, repo in (("ag_news", "fancyzhx/ag_news"), ("boolq", "google/boolq"))]


class DecisionHelpersTest(unittest.TestCase):
    def test_sample_is_seeded_and_stable(self):
        # Defined by sha256, so every Python version picks the same rows.
        self.assertEqual(cb.sample_indices(20, 5, 0, "ag_news"), [0, 2, 3, 13, 17])
        self.assertEqual(cb.sample_indices(20, 5, 1, "ag_news"), [0, 3, 4, 9, 17])
        self.assertEqual(cb.sample_indices(3, 10, 0, "x"), [0, 1, 2])

    def test_rows_hash_is_canonical(self):
        self.assertEqual(cb.rows_hash([{"a": 1, "b": "é"}]), cb.rows_hash([{"b": "é", "a": 1}]))
        self.assertNotEqual(cb.rows_hash([{"a": 1}, {"a": 2}]), cb.rows_hash([{"a": 2}, {"a": 1}]))

    def test_check_answers(self):
        qs = cb.latency_questions(3)
        good = {"topic": {"type": "choice", "choice": "nature"}, "formal": {"type": "noul", "noul": 0.3},
                "clarity": {"type": "score", "score": 1.2}}
        self.assertIsNone(cb.check_answers(qs, good))
        self.assertIn("no answer for 'clarity'", cb.check_answers(qs, {k: good[k] for k in ("topic", "formal")}))
        bad = dict(good, topic={"type": "choice", "choice": "sports"})
        self.assertIn("not a choice answer", cb.check_answers(qs, bad))
        self.assertEqual(cb.check_answers(qs, None), "no answers object in the response")

    def test_accuracy_request(self):
        state, q, truth = cb.accuracy_request(cb.ACCURACY_SETS["ag_news"], {"text": "Markets fell.", "label": 2})
        self.assertEqual((state, truth, q["type"]), ("Markets fell.", "Business", "choice"))
        self.assertEqual(list(q["criteria"]), ["World", "Sports", "Business", "Sci/Tech"])
        state, q, truth = cb.accuracy_request(cb.ACCURACY_SETS["boolq"],
                                              {"question": "is the sky blue", "passage": "It is.", "answer": True})
        self.assertEqual(q["instructions"], "Is the sky blue?")
        self.assertIs(truth, True)

    def test_engine_rejection_is_an_error(self):
        srv = fake_server.start(fake_server.FakeConfig(api_key="k"))
        try:
            ep = cb.Endpoint(srv.url, "k")
            rec = cb.post_decision(ep, {"model": "m", "state": "x", "questions": {"q": {"type": "noul"}}})
            self.assertTrue(rec["error"].startswith("HTTP 400"), rec["error"])
            self.assertIn('instructions\\" must be provided', rec["error"])
            rec = cb.post_decision(ep, {"model": "m", "state": "a b c", "questions": cb.latency_questions(5)},
                                   keep_answers=True)
            self.assertIsNone(rec["error"])
            self.assertEqual(rec["questions"], 5)
            self.assertEqual(set(rec["answers"]), {"topic", "formal", "clarity", "season", "question"})
            self.assertTrue(rec["extras"]["capyctl_bench_sample"])
            self.assertTrue(cb.post_decision(cb.Endpoint(srv.url, "wrong"), {
                "model": "m", "state": "x", "questions": cb.latency_questions(1)})["error"].startswith("HTTP 401"))
        finally:
            srv.shutdown()
            srv.server_close()

    def test_pinned_hash_mismatch_fails(self):
        srv = fake_server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.dict(cb.ACCURACY_SETS["boolq"], {"sha256": "0" * 64}), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    cb.load_dataset("boolq", srv.url[:-3], Path(tmp))
                self.assertIn("is not the pinned", str(cm.exception))
                self.assertEqual(list(Path(tmp).iterdir()), [])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_mode_options(self):
        with self.assertRaises(SystemExit) as cm:
            run_cli("run", "--model", "m", "--label", "x", "--concurrency", "1", "--accuracy")
        self.assertIn("--accuracy need --decisions", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            run_cli("run", "--model", "m", "--label", "x", "--decisions", "--context-sweep", "1k")
        self.assertIn("--context-sweep is for chat runs", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            run_cli("run", "--model", "m", "--label", "x", "--decisions", "--accuracy-samples", "5")
        self.assertIn("need --accuracy", str(cm.exception))


class DecisionsRunAndReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.cache = cls.dir / "datasets"
        cls.pins = fake_pins()
        for p in cls.pins:
            p.start()
        cls.results, cls.states, cls.rows_requests = [], [], []
        for label, base in (("Engine A (sample)", 0.003), ("Engine B (sample)", 0.005)):
            srv = fake_server.start(fake_server.FakeConfig(decision_base=base, api_key="secret"))
            try:
                root = srv.url[:-len("/v1")]
                mem = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
                    f"import urllib.request;print(urllib.request.urlopen('{root}/debug/memory').read().decode())")
                out = cls.dir / f"decisions-{label.split()[1].lower()}.json"
                key = cls.dir / "credentials"
                key.write_text("api_key: secret\n")
                rc = run_cli("run", "--decisions", "--endpoint", srv.url, "--api-key-file", key,
                             "--model", "sample-model", "--label", label, "--question-counts", "1,3",
                             "--state-sweep", "0.2k,2k", "--runs", "3", "--concurrency", "1,2,4", "--rounds", "2",
                             "--accuracy", "--accuracy-samples", "25", "--datasets-url", root,
                             "--dataset-cache", cls.cache, "--memory-cmd", mem, "--memory-interval", "0.05",
                             "--meta", "capyctl=0.0.0-test", "--meta", "gpu=fake", "--out", out)
                assert rc == 0
                cls.results.append(out)
                cls.states.append([r["state"] for r in srv.requests])
                cls.rows_requests.append(srv.rows_requests)
            finally:
                srv.shutdown()
                srv.server_close()

    @classmethod
    def tearDownClass(cls):
        for p in cls.pins:
            p.stop()
        cls.tmp.cleanup()

    def test_results_schema(self):
        data = json.loads(self.results[0].read_text())
        self.assertEqual(data["schema_version"], cb.SCHEMA_VERSION)
        self.assertEqual(data["mode"], "decisions")
        self.assertEqual(data["model"], "sample-model")
        self.assertEqual(data["meta"], {"capyctl": "0.0.0-test", "gpu": "fake", "sample": True})
        self.assertNotIn("context_sweep", data)
        dec = data["decisions"]
        self.assertEqual(dec["calibration"]["method"], "usage.input_tokens")
        pts = dec["latency"]["points"]
        self.assertEqual([p["label"] for p in pts], ["1q 200", "1q 2k", "3q 200", "3q 2k"])
        for p in pts:
            self.assertEqual(len(p["requests"]), 3)
            self.assertEqual(p["summary"]["errors"], 0)
            self.assertTrue(all(r["questions"] == p["questions"] for r in p["requests"]))
            self.assertLessEqual(p["summary"]["latency_p50_s"], p["summary"]["latency_p95_s"])
            self.assertIsNotNone(p["summary"]["memory_peak_gib"]["median"])
        # the state lengths come from calibration: input tokens grow about tenfold from 0.2k to 2k
        tok = {p["label"]: p["summary"]["input_tokens"]["median"] for p in pts}
        self.assertGreater(tok["1q 2k"], tok["1q 200"] * 5)
        self.assertGreater(tok["3q 200"], tok["1q 200"])
        levels = dec["concurrency"]["levels"]
        self.assertEqual([lv["concurrency"] for lv in levels], [1, 2, 4])
        for lv in levels:
            self.assertEqual(len(lv["rounds"]), 2)
            self.assertEqual(lv["rounds"][0]["decisions"], lv["concurrency"])
            self.assertTrue(all(r["questions"] == 3 for rnd in lv["rounds"] for r in rnd["requests"]))
            self.assertIsNotNone(lv["summary"]["latency_p95_s"])
        dps = [lv["summary"]["decisions_per_s"]["median"] for lv in levels]
        self.assertGreater(dps[2], dps[0])
        self.assertTrue(data["memory"]["samples"])

    def test_every_request_has_its_own_state(self):
        # Calibration, warm-ups, the latency sweep and every concurrency round (across stream
        # counts) send states no other request sent; only the accuracy rows are dataset text.
        n_latency = 2 + 1 + 4 * 3 + sum(c * 3 for c in (1, 2, 4))
        filler = self.states[0][:n_latency]
        self.assertEqual(len(set(filler)), len(filler))
        # the same run sends the same states (concurrent ones in any order): series are comparable
        self.assertEqual(sorted(self.states[0]), sorted(self.states[1]))

    def test_accuracy(self):
        data = json.loads(self.results[0].read_text())
        sets = {st["name"]: st for st in data["decisions"]["accuracy"]["sets"]}
        self.assertEqual(list(sets), ["ag_news", "boolq"])
        ag, bq = sets["ag_news"], sets["boolq"]
        self.assertEqual(ag["dataset"]["repo"], "fancyzhx/ag_news")
        self.assertEqual(ag["dataset"]["revision"], cb.ACCURACY_SETS["ag_news"]["revision"])
        self.assertEqual(ag["dataset"]["rows"], 60)
        self.assertEqual(ag["sample"], dict(ag["sample"], n=25, seed=0))
        self.assertEqual([i["row_idx"] for i in ag["items"]], cb.sample_indices(60, 25, 0, "ag_news"))
        # the fake names the wrong section in every fifth row
        for it in ag["items"]:
            self.assertEqual(it["correct"], it["row_idx"] % 5 != 0, it)
            self.assertEqual(it["prediction"] == it["label"], it["correct"])
            self.assertAlmostEqual(sum(it["probabilities"].values()), 1.0)
        want = sum(1 for it in ag["items"] if it["row_idx"] % 5) / 25
        self.assertAlmostEqual(ag["summary"]["accuracy"], want)
        self.assertIsNone(ag["summary"]["brier"])
        # the fake's passage gives the wrong answer in every fourth row; p(yes) is 0.9 or 0.2
        brier = 0.0
        for it in bq["items"]:
            self.assertEqual(it["correct"], it["row_idx"] % 4 != 3, it)
            says_yes = it["label"] if it["row_idx"] % 4 != 3 else not it["label"]
            self.assertEqual(it["noul"], 0.9 if says_yes else 0.2)
            brier += (it["noul"] - (1.0 if it["label"] else 0.0)) ** 2
        self.assertAlmostEqual(bq["summary"]["brier"], brier / 25)
        self.assertAlmostEqual(bq["summary"]["accuracy"],
                               sum(1 for it in bq["items"] if it["row_idx"] % 4 != 3) / 25)
        self.assertIsNotNone(bq["summary"]["memory_peak_gib"])

    def test_dataset_cache(self):
        # The first run downloads the sets once; the second reads them from the cache.
        self.assertEqual(self.rows_requests, [2, 0])
        self.assertEqual(sorted(p.name for p in self.cache.iterdir()),
                         ["ag_news-test-eb185aade064.jsonl", "boolq-validation-35b264d03638.jsonl"])

    def test_report(self):
        out = self.dir / "report"
        rc = run_cli("report", *self.results, "--out", out, "--title", "Fake decisions A vs B")
        self.assertEqual(rc, 0)
        page = (out / "report.html").read_text()
        for heading in ('<h2 id="latency">', '<h2 id="concurrency">', '<h2 id="accuracy">'):
            self.assertIn(heading, page)
        self.assertEqual(page.count('<svg class="chart"'), 4)  # 1 and 3 questions, decisions/s, latency
        self.assertIn("Dashed lines are p95.", page)
        self.assertIn("pt</small>", page)
        self.assertIn("/v1/systemone", page)
        self.assertNotIn("Temperature", page)
        self.assertIn("fancyzhx/ag_news test at eb185aade064", page)
        summary = (out / "summary.md").read_text()
        self.assertIn("| Point | Engine A (sample) | Engine B (sample) |", summary)
        self.assertIn("| 3 questions, 2k-token state |", summary)
        self.assertIn("### Decisions per second (decisions/s, higher is better)", summary)
        self.assertRegex(summary, r"\| AG News \| \d+\.\d% \| \d+\.\d% \([+-]\d+\.\d pt\) \|")
        self.assertIn("### Brier score (lower is better)", summary)
        self.assertIn("`google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`", summary)
        rows = (out / "data.csv").read_text().splitlines()
        self.assertEqual(rows[0], "series,section,point,x,metric,unit,median,min,max,n")
        self.assertEqual({r.split(",")[1] for r in rows[1:]}, {"latency", "concurrency", "accuracy"})
        self.assertTrue(any(r.startswith("Engine B (sample),accuracy,boolq,25,brier,") for r in rows))
        pngs = sorted(p.name for p in (out / "charts").glob("*.png")) if (out / "charts").exists() else []
        if HAVE_MPL:
            self.assertEqual(pngs, ["concurrency-decisions_per_s.png", "concurrency-latency.png",
                                    "latency-1q.png", "latency-3q.png"])
        else:
            self.assertEqual(pngs, [])

    def test_summary(self):
        out = self.dir / "shared"
        rc = run_cli("report", *self.results, "--out", out, "--summary", "--no-zip")
        self.assertEqual(rc, 0)
        page = (out / "summary" / "summary.html").read_text()
        self.assertIn('<p class="sample" role="note">SAMPLE DATA — not a measurement</p>', page)
        for title in ("Decisions per second", "Latency p50, 3 questions", "Decisions/s vs streams", "Accuracy"):
            self.assertIn(f"<h3>{title} <small>", page)
        self.assertIn("ms, lower is better", page)
        # throughput bars change in %, accuracy bars in percentage points
        self.assertRegex(page, r'class="barpct"[^>]*>[+-]\d+\.\d%<')
        self.assertRegex(page, r'class="barpct"[^>]*>[+-]\d+\.\d pt<')
        self.assertIn("accuracy on 25 rows each of AG News and BoolQ (seed 0)", page)
        names = sorted(p.name for p in (out / "summary").glob("*.png"))
        if HAVE_MPL:
            self.assertEqual(names, ["summary-tall-dark.png", "summary-tall.png",
                                     "summary-wide-dark.png", "summary-wide.png"])
        else:
            self.assertEqual(names, [])

    def test_latency_only_headline(self):
        # Without concurrency or accuracy the summary leads with latency p50 bars (lower is better).
        data = json.loads(self.results[0].read_text())
        data["decisions"]["concurrency"] = None
        data["decisions"]["accuracy"] = None
        lat = self.dir / "latency-only.json"
        lat.write_text(json.dumps(data))
        series = cb.load_series([lat])
        panels = cb.summary_panels(series)
        self.assertEqual([p["title"] for p in panels], ["Latency p50", "Latency p50, 3 questions"])
        self.assertEqual(panels[0]["better"], "lower")
        self.assertEqual([g["label"] for g in panels[0]["groups"]], ["1 question, 200 tok", "3 questions, 2k tok"])
        out = self.dir / "latency-only"
        self.assertEqual(run_cli("report", lat, "--out", out, "--summary", "--no-png", "--no-zip"), 0)
        self.assertNotIn('<h2 id="accuracy">', (out / "report.html").read_text())

    def test_chat_and_decisions_are_not_mixed(self):
        chat = json.loads(self.results[0].read_text())
        chat.pop("decisions")
        chat["mode"] = "chat"
        p = self.dir / "chat.json"
        p.write_text(json.dumps(chat))
        with self.assertRaises(SystemExit) as cm:
            cb.load_series([self.results[0], p])
        self.assertIn("report them separately", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
