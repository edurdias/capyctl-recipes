"""Unit tests for capyctl-bench. Run from tools/capyctl-bench:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import random
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
    """The datasets-server sets pinned to the fake server's copies of them."""
    return [mock.patch.dict(cb.ACCURACY_SETS[name], {"sha256": cb.rows_hash(fake_server.fake_rows(repo)[1])})
            for name, repo in (("ag_news", "fancyzhx/ag_news"), ("boolq", "google/boolq"),
                               ("typed_decisions", "LocalLLaMA/typed-decisions"))]


def fake_sources(root: str):
    """API-Bank and GPQA Diamond built from the fake server at `root`, pinned to what they build to.
    One of the six fake GPQA rows is excluded, as the kit excludes two real ones."""
    over = {"api_bank": {"tree_url": root + "/github/tree/{commit}", "raw_url": root + "/github/raw/{commit}/{path}"},
            "gpqa_diamond": {"source": dict(cb.ACCURACY_SETS["gpqa_diamond"]["source"], url=root + "/gpqa/dataset.zip",
                                            sha256=hashlib.sha256(fake_server.GPQA_ZIP).hexdigest()),
                             "excluded": ["GPQA-Diamond:test:4"]}}
    for name, keys in over.items():
        spec = dict(cb.ACCURACY_SETS[name], **keys)
        with contextlib.redirect_stderr(io.StringIO()):
            rows = cb.DI_BUILDERS[name](spec)
        keys.update(sha256=cb.sha256_hex(cb.di_serialize(rows, spec)), rows=len(rows))
    return [mock.patch.dict(cb.ACCURACY_SETS[name], keys) for name, keys in over.items()]


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
            run_cli("run", "--model", "m", "--label", "x", "--concurrency", "1", "--plan")
        self.assertIn("--plan need --decisions", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            run_cli("run", "--model", "m", "--label", "x", "--decisions", "--context-sweep", "1k")
        self.assertIn("--context-sweep is for chat runs", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            run_cli("run", "--model", "m", "--label", "x", "--decisions", "--rounds", "2")
        self.assertIn("--rounds is for chat runs", str(cm.exception))
        for opt in ("--accuracy-samples", "--accuracy-limit"):
            with self.assertRaises(SystemExit) as cm:
                run_cli("run", "--model", "m", "--label", "x", "--decisions", opt, "5")
            self.assertIn("need --accuracy", str(cm.exception))

    def test_suite_names(self):
        self.assertEqual(cb.parse_sets("typed"), ["typed_decisions"])
        self.assertEqual(cb.parse_sets("decision-index,boolq"), ["api_bank", "gpqa_diamond", "home_appliances", "boolq"])
        self.assertEqual(cb.parse_sets("classic,ag_news"), ["ag_news", "boolq"])
        self.assertEqual(cb.parse_sets("all"), list(cb.ACCURACY_SETS))
        with self.assertRaises(Exception):
            cb.parse_sets("mmlu")

    def test_plan_sends_nothing(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = run_cli("run", "--decisions", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m", "--label", "x",
                         "--accuracy", "--plan")
        self.assertEqual(rc, 0)
        text = out.getvalue()
        # 945 latency (9 points x 105), 206 cache, 823 throughput, 2 calibration, 2,192 quality rows
        for line in ("latency grid", "cache test", "throughput", "Typed Decisions", "API-Bank", "Home appliances"):
            self.assertIn(line, text)
        self.assertRegex(text, r"total\s+4,168 requests")
        args = cb.build_parser().parse_args(["run", "--decisions", "--model", "m", "--label", "x", "--accuracy",
                                             "--accuracy-sets", "typed,decision-index"])
        cb.check_mode_options(args)
        plan = cb.plan_decisions(args)
        self.assertEqual({s["section"]: s["requests"] for s in plan["sections"]
                          if s["section"] in ("Typed Decisions", "API-Bank", "GPQA Diamond", "Home appliances")},
                         {"Typed Decisions": 400, "API-Bank": 508, "GPQA Diamond": 196, "Home appliances": 88})


class ScoringTest(unittest.TestCase):
    def test_ece_bins(self):
        ece, bins = cb.ece_bins([(0.95, True), (0.95, False), (0.55, True), (1.0, True)])
        self.assertEqual(len(bins), 10)
        self.assertEqual([b["n"] for b in bins], [0, 0, 0, 0, 0, 1, 0, 0, 0, 3])
        top = bins[9]
        self.assertAlmostEqual(top["confidence"], (0.95 + 0.95 + 1.0) / 3)
        self.assertAlmostEqual(top["accuracy"], 2 / 3)
        self.assertAlmostEqual(ece, 3 / 4 * abs(2 / 3 - 2.9 / 3) + 1 / 4 * abs(1 - 0.55))
        self.assertEqual(cb.ece_bins([]), (None, [{"lo": k / 10, "hi": (k + 1) / 10, "n": 0, "confidence": None,
                                                   "accuracy": None} for k in range(10)]))

    def test_macro_f1(self):
        # a: tp 1, fp 1, fn 1 -> 0.5; b: tp 1, fp 0, fn 1 -> 2/3; c: tp 0, fp 1, fn 0 -> 0
        pairs = [("a", "a"), ("a", "c"), ("b", "a"), ("b", "b")]
        self.assertAlmostEqual(cb.macro_f1(pairs), (0.5 + 2 / 3 + 0) / 3)

    def test_classic_calibration(self):
        ag = cb.ACCURACY_SETS["ag_news"]
        items = [{"label": "World", "prediction": "World", "correct": True, "latency_s": 0.1, "error": None,
                  "probabilities": {"World": 0.7, "Sports": 0.1, "Business": 0.1, "Sci/Tech": 0.1}},
                 {"label": "Sports", "prediction": "World", "correct": False, "latency_s": 0.1, "error": None,
                  "probabilities": {"World": 0.5, "Sports": 0.5}}]
        sm = cb.summarize_accuracy(items, ag)
        self.assertAlmostEqual(sm["brier_vector"], ((0.3 ** 2 + 3 * 0.01) + (0.25 + 0.25)) / 2)
        self.assertAlmostEqual(sm["nll"], (-math.log(0.7) - math.log(0.5)) / 2)
        self.assertIsNone(sm["brier"])
        # the tie between World and Sports goes to World, the first label: wrong at 0.5
        self.assertAlmostEqual(sm["ece"], 0.5 * abs(1 - 0.7) + 0.5 * abs(0 - 0.5))
        bq = cb.ACCURACY_SETS["boolq"]
        items = [{"label": True, "noul": 0.8, "prediction": True, "correct": True, "latency_s": 0.1, "error": None},
                 {"label": False, "noul": 0.6, "prediction": True, "correct": False, "latency_s": 0.1, "error": None}]
        sm = cb.summarize_accuracy(items, bq)
        self.assertAlmostEqual(sm["brier"], (0.04 + 0.36) / 2)
        self.assertAlmostEqual(sm["brier_vector"], 2 * sm["brier"])
        self.assertAlmostEqual(sm["nll"], (-math.log(0.8) - math.log(0.4)) / 2)
        self.assertEqual(sm["nll_eps"], 1e-9)

    def test_typed_decision(self):
        gold = {"type": "choice", "label": "a", "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}}
        d = cb.score_typed_decision("q", gold, {"type": "choice", "choice": "b",
                                                "probabilities": {"a": 0.2, "b": 0.7, "c": 0.1}})
        self.assertEqual((d["prediction"], d["correct"], d["correct_gold_ties"]), ("b", False, False))
        self.assertAlmostEqual(d["kl"], 0.5 * math.log(0.5 / 0.2) + 0.3 * math.log(0.3 / 0.7) + 0.2 * math.log(2),
                               places=6)
        self.assertAlmostEqual(d["brier"], 0.09 + 0.16 + 0.01)
        self.assertAlmostEqual(d["brier_hard"], 0.64 + 0.49 + 0.01)
        self.assertEqual(d["confidence"], 0.7)
        # a uniform answer against a one-hot gold: KL is ln K (the clip only matters at p = 0)
        hot = {"type": "choice", "label": "a", "probabilities": {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}}
        d = cb.score_typed_decision("q", hot, {"choice": "a", "probabilities": {k: 0.25 for k in "abcd"}})
        self.assertAlmostEqual(d["kl"], math.log(4), places=6)
        self.assertTrue(d["correct"])  # a four-way tie goes to the first option
        d = cb.score_typed_decision("q", hot, {"choice": "b", "probabilities": {"b": 1.0}})
        self.assertAlmostEqual(d["kl"], -math.log(1e-9), places=4)
        # a noul answer carries only p(true); a choice without probabilities is one-hot (derived)
        nl = {"type": "noul", "label": "true", "probabilities": {"false": 0.4, "true": 0.6}}
        d = cb.score_typed_decision("q", nl, {"type": "noul", "noul": 0.9})
        self.assertEqual((d["probabilities"], d["derived"], d["correct"]), ({"false": 0.1, "true": 0.9}, False, True))
        d = cb.score_typed_decision("q", gold, {"type": "choice", "choice": "c"})
        self.assertEqual((d["probabilities"], d["derived"]), ({"a": 0.0, "b": 0.0, "c": 1.0}, True))
        sc = {"type": "score", "label": "1", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}}
        d = cb.score_typed_decision("q", sc, {"type": "score", "score": 1.5, "probabilities": {"0": 0, "1": 0.5, "2": 0.5}})
        self.assertAlmostEqual(d["score_error"], 0.3)
        self.assertEqual(d["prediction"], "1")  # tie between 1 and 2: the first in the gold's order

    def test_index_value_matches_the_board(self):
        # Values of the decision-index kit's tests/fixtures/board-0.3.json (raw, skill to 4 places)
        for score, answered, requests, chance, raw, skill in (
                (154 / 196, 196, 196, 0.25, 0.7857, 0.7143),   # Jev, GPQA Diamond
                (467 / 508, 508, 508, 0.0189, 0.9193, 0.9177),  # Cloudflare clef, API-Bank
                (46 / 88, 88, 88, 0.0, 0.5227, 0.5227),        # Jev, Home appliances
                (None, 0, 508, 0.0189, 0.0, 0.0)):             # Verdict, API-Bank: nothing answered
            v = cb.index_value(score, answered, requests, chance)
            self.assertAlmostEqual(v["raw"], raw, places=4)
            self.assertAlmostEqual(round(v["skill"], 4), skill, places=4)
        self.assertAlmostEqual(cb.index_value(0.5, 254, 508, 0.25)["raw"], 0.25)

    def test_rank_shades(self):
        # higher is better: best 1, worst 0; values that print the same tie; no value, no shade
        self.assertEqual(cb.rank_shades(["70.0", "90.0", None, "80.0", "90.0"], "higher"), [0.0, 1.0, None, 0.5, 1.0])
        self.assertEqual(cb.rank_shades(["2,202", "56.0", "551"], "lower"), [0.0, 1.0, 0.5])
        self.assertEqual(cb.rank_shades(["1.00"], "lower"), [1.0])

    def test_place_labels_never_overlap(self):
        # Five points almost on top of each other, labels 60 x 12: every label clear of the others,
        # the markers and the bounds; leaders (needed here) cross no label, marker or other leader.
        anchors = [(100, 100), (102, 101), (104, 99), (100, 103), (300, 20)]
        placed = cb.place_labels(anchors, [(60, 12)] * 5, (0, 0, 400, 200), marker=5)
        markers = [(x - 5, y - 5, x + 5, y + 5) for x, y in anchors]
        for i, (a, end) in enumerate(placed):
            self.assertTrue(0 <= a[0] and a[2] <= 400 and 0 <= a[1] and a[3] <= 200, a)
            self.assertFalse(any(cb._overlap(a, m) for m in markers), a)
            for j, (b, other) in enumerate(placed):
                if j == i:
                    continue
                self.assertFalse(cb._overlap(a, b), (a, b))
                if end is not None:
                    self.assertFalse(cb._seg_hits_box(end, anchors[i], b))
                    if other is not None:
                        self.assertFalse(cb._segs_cross(end, anchors[i], other, anchors[j]))

    def test_place_labels_beside_their_own_marker(self):
        # Room around every point: each label touches the side of its own marker, no leader, and is
        # nearer its own marker than any other.
        anchors = [(100, 100), (160, 104), (130, 60), (300, 150), (60, 170)]
        placed = cb.place_labels(anchors, [(70, 12)] * 5, (0, 0, 400, 200), marker=5)
        for i, (box, end) in enumerate(placed):
            self.assertIsNone(end)
            own = cb._box_dist(anchors[i], box)
            self.assertLessEqual(own, 5 + 3 + 1)
            self.assertTrue(all(cb._box_dist(anchors[j], box) > own for j in range(5) if j != i))

    def test_failure_status(self):
        self.assertEqual(cb.failure_status('HTTP 400: {"error": "the request exceeds the maximum context length"}'),
                         "unsupported")
        self.assertEqual(cb.failure_status("HTTP 422: a choice needs at least two options"), "unsupported")
        self.assertEqual(cb.failure_status("HTTP 500: maximum context length"), "error")
        self.assertEqual(cb.failure_status("TimeoutError: timed out"), "error")


class DecisionIndexBuildTest(unittest.TestCase):
    def test_home_appliances_reproduce_the_kit(self):
        # Generated, no download: the rows must hash to the kit's pin and its 0.2.1 cut keep 88.
        with tempfile.TemporaryDirectory() as tmp:
            built, kept = cb.load_di_rows("home_appliances", Path(tmp))
            self.assertEqual((len(built), len(kept)), (160, 88))
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["home_appliances-13862aa65d24.jsonl"])
            self.assertEqual(kept[0]["id"], "home-appliance:test:indigo:00")
            self.assertEqual(sum(len(r["questions"]) for r in kept), 1590)
            self.assertEqual(cb.di_run_id(cb.ACCURACY_SETS["home_appliances"], kept[0]),
                             "9:Home-Appliance:home-appliance:test:indigo:00")
            # the second load reads the cache
            with mock.patch.dict(cb.DI_BUILDERS, {"home_appliances": None}):
                self.assertEqual(len(cb.load_di_rows("home_appliances", Path(tmp))[1]), 88)

    def test_rebuilt_rows_must_match_the_pin(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(cb.ACCURACY_SETS["home_appliances"], {"sha256": "0" * 64}):
            with self.assertRaises(SystemExit) as cm:
                cb.load_di_rows("home_appliances", Path(tmp))
            self.assertIn("an upstream source changed", str(cm.exception))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_api_bank_and_gpqa_builders(self):
        srv = fake_server.start()
        try:
            root = srv.url[:-3]
            spec = dict(cb.ACCURACY_SETS["api_bank"], tree_url=root + "/github/tree/{commit}",
                        raw_url=root + "/github/raw/{commit}/{path}")
            with contextlib.redirect_stderr(io.StringIO()):
                rows = cb.build_api_bank(spec)
            # three API turns, the ToolSearcher one dropped: its tool is not in the catalog
            self.assertEqual([r["id"] for r in rows], [
                "API-Bank-tool-selection:AddAlarm-level-1-1:api-1", "API-Bank-tool-selection:AddAlarm-level-1-1:api-4",
                "API-Bank-tool-selection:QueryStock-level-2-1:api-2"])
            r = rows[1]
            self.assertEqual([t["name"] for t in r["state"]["available_tools"]], ["AddAlarm", "GetWeather", "QueryStock"])
            self.assertEqual(len(r["state"]["dialogue"]), 4)
            crit = r["questions"]["tool"]["criteria"]
            self.assertEqual(sorted(crit.values()), ["AddAlarm", "GetWeather", "QueryStock"])
            self.assertEqual(crit[r["expected"]["tool"]], "GetWeather")
            order = list(range(3))
            random.Random("20260918:AddAlarm-level-1-1.jsonl:4").shuffle(order)
            self.assertEqual(list(crit.values()), [["AddAlarm", "GetWeather", "QueryStock"][j] for j in order])
            self.assertEqual(r["state"]["available_tools"][0]["description"], "Set an alarm")
            spec = dict(cb.ACCURACY_SETS["gpqa_diamond"])
            spec["source"] = dict(spec["source"], url=root + "/gpqa/dataset.zip",
                                  sha256=hashlib.sha256(fake_server.GPQA_ZIP).hexdigest())
            with contextlib.redirect_stderr(io.StringIO()):
                rows = cb.build_gpqa_diamond(spec)
            self.assertEqual(len(rows), 6)
            for i, r in enumerate(rows):
                q = r["questions"]["answer"]
                self.assertEqual((r["state"], q["instructions"]), ("", f"Question {i}: which is right?"))
                self.assertEqual(list(q["criteria"]), ["A", "B", "C", "D"])
                self.assertEqual(q["criteria"][r["expected"]["answer"]], f"right {i}")
            bad = dict(spec, source=dict(spec["source"], sha256="0" * 64))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cb.build_gpqa_diamond(bad)
        finally:
            srv.shutdown()
            srv.server_close()


class DecisionsRunAndReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.cache = cls.dir / "datasets"
        # one fake server for the dataset sources, one per engine
        cls.src = fake_server.start()
        root = cls.src.url[:-len("/v1")]
        cls.pins = fake_pins() + fake_sources(root)
        for p in cls.pins:
            p.start()
        cls.src.rows_requests = cls.src.source_requests = 0
        cls.results, cls.requests, cls.downloads = [], [], []
        for label, base in (("Engine A (sample)", 0.003), ("Engine B (sample)", 0.005)):
            srv = fake_server.start(fake_server.FakeConfig(decision_base=base, decision_per_token=4e-6,
                                                           api_key="secret"))
            try:
                eroot = srv.url[:-len("/v1")]
                mem = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
                    f"import urllib.request;print(urllib.request.urlopen('{eroot}/debug/memory').read().decode())")
                out = cls.dir / f"decisions-{label.split()[1].lower()}.json"
                key = cls.dir / "credentials"
                key.write_text("api_key: secret\n")
                rc = run_cli("run", "--decisions", "--endpoint", srv.url, "--api-key-file", key,
                             "--model", "sample-model", "--label", label, "--question-counts", "1,3",
                             "--state-sweep", "0.2k,2k", "--runs", "3", "--warmup", "1", "--cache-runs", "4",
                             "--concurrency", "1,2,4", "--throughput-requests", "12",
                             "--accuracy", "--accuracy-samples", "25", "--datasets-url", root,
                             "--dataset-cache", cls.cache, "--memory-cmd", mem, "--memory-interval", "0.05",
                             "--meta", "capyctl=0.0.0-test", "--meta", "gpu=fake", "--out", out)
                assert rc == 0
                cls.results.append(out)
                cls.requests.append(list(srv.requests))
                cls.downloads.append((cls.src.rows_requests, cls.src.source_requests))
            finally:
                srv.shutdown()
                srv.server_close()

    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.pins):
            p.stop()
        cls.src.shutdown()
        cls.src.server_close()
        cls.tmp.cleanup()

    def data(self, i: int = 0) -> dict:
        return json.loads(self.results[i].read_text())

    def sets(self, i: int = 0) -> dict:
        return {st["name"]: st for st in self.data(i)["decisions"]["accuracy"]["sets"]}

    def test_results_schema(self):
        data = self.data()
        self.assertEqual(data["schema_version"], cb.SCHEMA_VERSION)
        self.assertEqual(data["mode"], "decisions")
        self.assertEqual(data["model"], "sample-model")
        self.assertEqual(data["meta"], {"capyctl": "0.0.0-test", "gpu": "fake", "sample": True})
        self.assertNotIn("context_sweep", data)
        dec = data["decisions"]
        man = dec["manifest"]
        self.assertEqual(man["question_order"], ["topic", "formal", "clarity", "season", "question"])
        self.assertEqual(man["question_types"]["clarity"], "score")
        self.assertEqual(man["options"], {"topic": 4, "formal": 2, "clarity": 3, "season": 3, "question": 2})
        self.assertEqual(dec["calibration"]["method"], "usage.input_tokens")
        pts = dec["latency"]["points"]
        self.assertEqual([p["label"] for p in pts], ["1q 200", "1q 2k", "3q 200", "3q 2k"])
        for p in pts:
            sm = p["summary"]
            self.assertEqual(len(p["requests"]), 3)
            self.assertEqual(sm["errors"], 0)
            self.assertTrue(all(r["questions"] == p["questions"] for r in p["requests"]))
            self.assertLessEqual(sm["latency_p50_s"], sm["latency_p95_s"])
            # amortized per answered question
            self.assertAlmostEqual(sm["per_question_p50_s"], sm["latency_p50_s"] / p["questions"])
            self.assertIsNotNone(sm["memory_peak_gib"]["median"])
            self.assertTrue(all(r["input_tokens"] for r in p["requests"]))
        # the state lengths come from calibration: input tokens grow about tenfold from 0.2k to 2k
        tok = {p["label"]: p["summary"]["input_tokens"]["median"] for p in pts}
        self.assertGreater(tok["1q 2k"], tok["1q 200"] * 5)
        self.assertGreater(tok["3q 200"], tok["1q 200"])
        self.assertEqual(dec["latency"]["settings"]["warmup"], 1)
        self.assertTrue(data["memory"]["samples"])

    def test_cache_test(self):
        cache = self.data()["decisions"]["cache"]
        self.assertEqual(cache["settings"]["runs"], 4)
        cold, rep = cache["cold"], cache["repeated"]
        self.assertEqual((len(cold["requests"]), len(rep["requests"])), (4, 4))
        self.assertEqual(cold["summary"]["errors"] + rep["summary"]["errors"], 0)
        # the fake server answers a state it has seen before faster: a prefix cache
        self.assertLess(rep["summary"]["latency_p50_s"], cold["summary"]["latency_p50_s"])
        self.assertEqual(rep["first"]["questions"], 3)

    def test_throughput(self):
        levels = self.data()["decisions"]["concurrency"]["levels"]
        self.assertEqual([lv["concurrency"] for lv in levels], [1, 2, 4])
        for lv in levels:
            sm = lv["summary"]
            self.assertEqual(len(lv["requests"]), 12)
            self.assertEqual({r["stream"] for r in lv["requests"]} <= set(range(lv["concurrency"])), True)
            self.assertEqual(sm["error_rate"], 0)
            self.assertAlmostEqual(sm["questions_per_s"], 3 * sm["requests_per_s"])
            self.assertAlmostEqual(sm["requests_per_s"], 12 / sm["wall_s"])
            self.assertIsNotNone(sm["latency_p95_s"])
        self.assertGreater(levels[2]["summary"]["requests_per_s"], levels[0]["summary"]["requests_per_s"])
        self.assertEqual(len({r["stream"] for r in levels[2]["requests"]}), 4)

    def test_every_request_has_its_own_state(self):
        # Calibration, warm-ups, the latency grid, the cache test and the throughput levels send states no
        # other request sent, but the cache test's repeated block; then come the dataset rows.
        n_speed = 2 + 4 * (1 + 3) + (1 + 4 + 1 + 4) + sum(max(c, 1) + 12 for c in (1, 2, 4))
        filler = [r["state"] for r in self.requests[0][:n_speed]]
        self.assertTrue(all(isinstance(s, str) for s in filler))
        self.assertEqual(len(set(filler)), len(filler) - 4)
        self.assertEqual(len(self.requests[0]), n_speed + 25 + 25 + 12 + 3 + 5 + 88)
        # the same run sends the same requests (concurrent ones in any order): series are comparable
        self.assertEqual(sorted(json.dumps(r, sort_keys=True) for r in self.requests[0]),
                         sorted(json.dumps(r, sort_keys=True) for r in self.requests[1]))

    def test_classic_sets(self):
        sets = self.sets()
        self.assertEqual(list(sets), list(cb.ACCURACY_SETS))
        ag, bq = sets["ag_news"], sets["boolq"]
        self.assertEqual((ag["kind"], ag["dataset"]["repo"]), ("classic", "fancyzhx/ag_news"))
        self.assertEqual(ag["dataset"]["revision"], cb.ACCURACY_SETS["ag_news"]["revision"])
        self.assertEqual(ag["dataset"]["rows"], 60)
        self.assertEqual(ag["sample"], dict(ag["sample"], n=25, seed=0))
        self.assertEqual([i["row_idx"] for i in ag["items"]], cb.sample_indices(60, 25, 0, "ag_news"))
        # the fake names the wrong section in every fifth row
        for it in ag["items"]:
            self.assertEqual(it["correct"], it["row_idx"] % 5 != 0, it)
            self.assertEqual(it["prediction"] == it["label"], it["correct"])
            self.assertAlmostEqual(sum(it["probabilities"].values()), 1.0)
        sm = ag["summary"]
        want = sum(1 for it in ag["items"] if it["row_idx"] % 5) / 25
        self.assertAlmostEqual(sm["accuracy"], want)
        self.assertIsNone(sm["brier"])
        # every answer puts 0.7 on its pick and 0.1 on the rest: one reliability bin
        self.assertEqual([b["n"] for b in sm["ece_bins"]], [0] * 7 + [25, 0, 0])
        self.assertAlmostEqual(sm["ece"], abs(want - 0.7))
        self.assertAlmostEqual(sm["brier_vector"], want * (0.09 + 0.03) + (1 - want) * (0.49 + 0.81 + 0.02))
        self.assertAlmostEqual(sm["nll"], -(want * math.log(0.7) + (1 - want) * math.log(0.1)))
        self.assertIsNotNone(sm["macro_f1"])
        # the fake's passage gives the wrong answer in every fourth row; p(yes) is 0.9 or 0.2
        brier = 0.0
        for it in bq["items"]:
            self.assertEqual(it["correct"], it["row_idx"] % 4 != 3, it)
            says_yes = it["label"] if it["row_idx"] % 4 != 3 else not it["label"]
            self.assertEqual(it["noul"], 0.9 if says_yes else 0.2)
            brier += (it["noul"] - (1.0 if it["label"] else 0.0)) ** 2
        self.assertAlmostEqual(bq["summary"]["brier"], brier / 25)
        self.assertAlmostEqual(bq["summary"]["brier_vector"], 2 * brier / 25)
        self.assertAlmostEqual(bq["summary"]["accuracy"],
                               sum(1 for it in bq["items"] if it["row_idx"] % 4 != 3) / 25)
        self.assertEqual(sum(b["n"] for b in bq["summary"]["ece_bins"]), 25)
        self.assertIsNotNone(bq["summary"]["memory_peak_gib"])

    def test_typed_decisions(self):
        st = self.sets()["typed_decisions"]
        rows = fake_server.typed_rows()
        self.assertEqual((st["kind"], st["rows"]), ("typed", {"n": 12, "of": 12, "partial": False}))
        self.assertEqual(st["dataset"]["revision"], "7189f46b02248bd2569df7325c9df8e3b24e90a7")
        # one request per case: the row's parsed state and all five questions
        sent = [r for r in self.requests[0] if isinstance(r["state"], dict) and "thread" in r["state"]]
        self.assertEqual([r["state"] for r in sent], [json.loads(r["state"]) for r in rows])
        self.assertEqual([r["questions"] for r in sent], [json.loads(r["questions"]) for r in rows])
        decisions = []
        for row, it in zip(rows, st["items"]):
            state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
            want = [cb.score_typed_decision(q, gold[q], fake_server.answer(q, questions[q], json.dumps(state)))
                    for q in gold]
            self.assertEqual(it["decisions"], want)
            self.assertEqual(it["payload_sha256"], hashlib.sha256(cb.canonical_row(
                {"state": state, "questions": questions}).encode()).hexdigest())
            # the state names the gold route in every case but the third ones
            self.assertEqual(it["decisions"][0]["correct"], int(row["id"][-6:]) % 3 != 2)
            decisions += want
        sm = st["summary"]
        self.assertEqual((sm["cases"], sm["answered"], sm["decisions"], sm["coverage"]), (12, 12, 60, 1.0))
        self.assertAlmostEqual(sm["accuracy"], sum(d["correct"] for d in decisions) / 60)
        self.assertAlmostEqual(sm["kl"], sum(d["kl"] for d in decisions) / 60)
        self.assertAlmostEqual(sm["brier_hard"], sum(d["brier_hard"] for d in decisions) / 60)
        self.assertEqual(sm["by_type"]["choice"]["decisions"], 24)
        self.assertEqual(sorted(sm["by_workflow"]), ["customer_service", "invoice_processing"])
        self.assertEqual(sum(b["n"] for b in sm["ece_bins"]), 60)
        self.assertIsNotNone(sm["score_mae"])
        self.assertEqual(sm["derived_probabilities"], 0)
        # `action`'s gold is a 50/50 tie: either answer is right counting gold ties
        self.assertTrue(all(d["correct_gold_ties"] for it in st["items"] for d in it["decisions"] if d["qid"] == "action"))

    def test_decision_index_components(self):
        sets = self.sets()
        api, gpqa, home = sets["api_bank"], sets["gpqa_diamond"], sets["home_appliances"]
        self.assertEqual([st["kind"] for st in (api, gpqa, home)], ["di"] * 3)
        self.assertEqual((api["dataset"]["kept"], gpqa["dataset"]["kept"], home["dataset"]["kept"]), (3, 5, 88))
        self.assertEqual((gpqa["dataset"]["rows"], home["dataset"]["rows"]), (6, 160))
        self.assertNotIn("password", gpqa["dataset"]["source"])
        self.assertNotIn("GPQA-Diamond:test:4", [i["id"] for i in gpqa["items"]])
        self.assertEqual(home["dataset"]["sha256"], cb.ACCURACY_SETS["home_appliances"]["sha256"])
        self.assertEqual(home["dataset"]["kit"]["commit"], cb.DI_KIT["commit"])
        # the kit's rows go out unchanged, and are scored on the answer's choice
        _, kept = cb.load_di_rows("home_appliances", self.cache)
        by_q = {json.dumps(r["questions"], sort_keys=True): r for r in self.requests[0]}
        for row, it in zip(kept, home["items"]):
            sent = by_q[json.dumps(row["questions"], sort_keys=True)]
            self.assertEqual(sent["state"], row["state"])
            hits = [fake_server.answer(k, q, json.dumps(row["state"]))["choice"] == row["expected"][k]
                    for k, q in row["questions"].items()]
            self.assertEqual((it["fields"], it["fields_correct"], it["correct"]), (len(hits), sum(hits), all(hits)))
            self.assertEqual(it["payload_sha256"], cb.di_payload_sha256(row))
            self.assertEqual(it["run_id"], "9:Home-Appliance:" + row["id"])
        for st in (api, gpqa, home):
            sm = st["summary"]
            items = st["items"]
            self.assertEqual((sm["requests"], sm["answered"], sm["unsupported"], sm["errors"]), (len(items), len(items), 0, 0))
            exact = sum(i["correct"] for i in items) / len(items)
            self.assertAlmostEqual(sm["case_exact_accuracy"], exact)
            self.assertAlmostEqual(sm["score"], exact if st["name"] == "home_appliances" else sm["field_accuracy"])
            self.assertAlmostEqual(sm["raw"], round(sm["score"], 4))
            chance = cb.ACCURACY_SETS[st["name"]]["chance"]
            self.assertAlmostEqual(sm["skill"], max(0.0, (sm["raw"] - chance) / (1 - chance)))
        self.assertEqual(home["summary"]["metric"], "case exact accuracy")
        self.assertEqual(sum(i["fields"] for i in home["items"]), 1590)
        self.assertIn("resolution", home["summary"]["field_accuracy_by_role"])
        self.assertEqual(home["request"]["questions_per_request"], {"min": 17, "max": 19})
        self.assertEqual(api["request"]["options_per_question"], {"min": 3, "max": 3})

    def test_dataset_cache(self):
        # The first run downloads and builds the sets once; the second reads them from the cache.
        (rows1, src1), (rows2, src2) = self.downloads
        self.assertEqual(rows1, 3)  # AG News, BoolQ, Typed Decisions: one page each
        self.assertEqual(src1, 1 + 7 + 1)  # API-Bank tree and its 7 files, the GPQA zip
        self.assertEqual((rows2, src2), (rows1, src1))
        names = sorted(p.name for p in self.cache.iterdir())
        self.assertEqual(names[:2], ["ag_news-test-eb185aade064.jsonl", "api_bank-" +
                                     cb.ACCURACY_SETS["api_bank"]["sha256"][:12] + ".jsonl"])
        self.assertIn("typed_decisions-test-7189f46b0224.jsonl", names)
        self.assertIn("home_appliances-13862aa65d24.jsonl", names)
        self.assertEqual(len(names), 6)

    def test_accuracy_limit(self):
        srv = fake_server.start()
        try:
            out = self.dir / "limited.json"
            rc = run_cli("run", "--decisions", "--endpoint", srv.url, "--model", "m", "--label", "x",
                         "--question-counts", "1", "--state-sweep", "0.2k", "--runs", "1", "--warmup", "0",
                         "--cache-runs", "0", "--throughput-requests", "0", "--accuracy",
                         "--accuracy-sets", "typed,home_appliances", "--accuracy-limit", "5",
                         "--datasets-url", self.src.url[:-3], "--dataset-cache", self.cache, "--out", out)
            self.assertEqual(rc, 0)
            data = json.loads(out.read_text())
            self.assertIsNone(data["decisions"]["cache"])
            self.assertIsNone(data["decisions"]["concurrency"])
            sets = data["decisions"]["accuracy"]["sets"]
            self.assertEqual([st["rows"] for st in sets], [{"n": 5, "of": 12, "partial": True},
                                                           {"n": 5, "of": 88, "partial": True}])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_report(self):
        out = self.dir / "report"
        rc = run_cli("report", *self.results, "--out", out, "--title", "Fake decisions A vs B")
        self.assertEqual(rc, 0)
        page = (out / "report.html").read_text()
        for heading in ('<h2 id="latency">', '<h2 id="cache">', '<h2 id="concurrency">', '<h2 id="accuracy">',
                        '<h2 id="calibration">', "<h3>Typed Decisions</h3>", "<h3>Decision Index components</h3>",
                        "<h3>Accuracy and calibration</h3>"):
            self.assertIn(heading, page)
        # 1 and 3 questions; requests/s, questions/s, latency; reliability of AG News, BoolQ, Typed Decisions
        self.assertEqual(page.count('<svg class="chart"'), 8)
        self.assertIn('class="ln diag"', page)
        self.assertIn("closer to the diagonal is better", page)
        self.assertIn("Dashed lines are p95.", page)
        self.assertIn("pt</small>", page)
        self.assertIn("/v1/systemone", page)
        self.assertNotIn("Temperature", page)
        self.assertIn("fancyzhx/ag_news test at eb185aade064", page)
        self.assertIn("Decision Index 9, kit f5a6e4fd85fe, 88 of 88 rows", page)
        summary = (out / "summary.md").read_text()
        self.assertIn("| Point | Engine A (sample) | Engine B (sample) |", summary)
        self.assertIn("| 3 questions, 2k-token state |", summary)
        self.assertIn("### Per answered question, p50 (ms, lower is better)", summary)
        self.assertIn("## Cache", summary)
        self.assertIn("| Repeated p50 / cold p50 |", summary)
        self.assertIn("### Requests per second (requests/s, higher is better)", summary)
        self.assertIn("### Answered questions per second (questions/s, higher is better)", summary)
        self.assertIn("## Typed Decisions", summary)
        self.assertRegex(summary, r"\| KL from gold \(lower is better\) \| \d\.\d+ \| \d\.\d+ \([+-]")
        self.assertRegex(summary, r"\| Accuracy \(%, higher is better\) \| \d+\.\d% \| \d+\.\d% \([+-]\d+\.\d pt\) \|")
        self.assertIn("### Typed Decisions accuracy by workflow (%, higher is better)", summary)
        self.assertIn("## Decision Index components", summary)
        self.assertRegex(summary, r"\| API-Bank \| \d+\.\d% \| \d+\.\d% \([+-]\d+\.\d pt\) \|")
        self.assertIn("#### Reliability, AG News", summary)
        self.assertRegex(summary, r"\| 0\.7–0\.8 \| \d+\.\d% at 70\.0% \(n 25\) \|")
        self.assertIn("`google/boolq` validation at revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`", summary)
        self.assertIn("`LocalLLaMA/typed-decisions` test at revision `7189f46b02248bd2569df7325c9df8e3b24e90a7`",
                      summary)
        self.assertIn("Home appliances: Decision Index catalog 9 (Home-Appliance)", summary)
        rows = (out / "data.csv").read_text().splitlines()
        self.assertEqual(rows[0], "series,section,point,x,metric,unit,median,min,max,n")
        self.assertEqual({r.split(",")[1] for r in rows[1:]},
                         {"latency", "cache", "concurrency", "accuracy", "reliability"})
        self.assertTrue(any(r.startswith("Engine B (sample),accuracy,boolq,25,brier,") for r in rows))
        self.assertTrue(any(r.startswith("Engine A (sample),accuracy,typed_decisions,12,kl,") for r in rows))
        self.assertTrue(any(r.startswith("Engine A (sample),accuracy,home_appliances,88,skill,%,") for r in rows))
        pngs = sorted(p.name for p in (out / "charts").glob("*.png")) if (out / "charts").exists() else []
        if HAVE_MPL:
            self.assertEqual(pngs, ["concurrency-latency.png", "concurrency-questions_per_s.png",
                                    "concurrency-requests_per_s.png", "latency-1q.png", "latency-3q.png",
                                    "reliability-ag_news.png", "reliability-boolq.png",
                                    "reliability-typed_decisions.png"])
        else:
            self.assertEqual(pngs, [])

    def test_summary(self):
        out = self.dir / "shared"
        rc = run_cli("report", *self.results, "--out", out, "--summary", "--no-zip")
        self.assertEqual(rc, 0)
        page = (out / "summary" / "summary.html").read_text()
        self.assertIn('<p class="sample" role="note">SAMPLE DATA — not a measurement</p>', page)
        for title in ("Accuracy", "Decision Index components", "Latency p50, 3 questions", "Questions/s vs streams"):
            self.assertIn(f"<h3>{title} <small>", page)
        self.assertIn("ms, lower is better", page)
        # quality bars change in percentage points
        self.assertRegex(page, r'class="barpct"[^>]*>[+-]\d+\.\d pt<')
        self.assertIn("accuracy on 25 rows each of AG News and BoolQ (seed 0)", page)
        self.assertIn("Typed Decisions 12 cases", page)
        self.assertIn("components (not an index) API-Bank 3, GPQA Diamond 5, Home appliances 88", page)
        names = sorted(p.name for p in (out / "summary").glob("*.png"))
        if HAVE_MPL:
            self.assertEqual(names, ["summary-tall-dark.png", "summary-tall.png",
                                     "summary-wide-dark.png", "summary-wide.png"])
        else:
            self.assertEqual(names, [])

    def field(self, n: int, models: bool = False) -> list[Path]:
        """n copies of the two runs, relabelled (and with a model of their own when `models`)."""
        paths = []
        for i in range(n):
            data = self.data(i % 2)
            data["label"] = f"Model {i} (sample)"
            if models:
                data["model"] = f"model-{i}"
            p = self.dir / f"field-{n}-{models}-{i}.json"
            p.write_text(json.dumps(data))
            paths.append(p)
        return paths

    def test_layout_selection(self):
        # Up to three series of one model: the bars. More, or different models: the scoreboard layout.
        self.assertFalse(cb.many_series(cb.load_series(self.results)))
        self.assertFalse(cb.many_series(cb.load_series(self.field(3))))
        self.assertTrue(cb.many_series(cb.load_series(self.field(4))))
        self.assertTrue(cb.many_series(cb.load_series(self.field(2, models=True))))
        self.assertEqual([p["kind"] for p in cb.summary_panels(cb.load_series(self.field(3)))], ["bars"] * 2 + ["line"] * 2)
        series = cb.load_series(self.field(6))
        panels = cb.summary_panels(series)
        self.assertEqual([p["kind"] for p in panels], ["scoreboard", "scatter"])
        board, scatter = panels
        self.assertEqual([c["head"][0] for c in board["cols"]],
                         ["Typed", "API-Bank", "GPQA", "Home", "AG News", "BoolQ", "Typed", "Latency", "Questions/s",
                          "Memory"])
        self.assertEqual(board["cols"][7]["head"][1], "1q 2k")
        self.assertEqual(board["cols"][8]["head"][2], "3q 2k")
        self.assertEqual(len(board["rows"]), 6)
        for c in board["cols"]:
            self.assertEqual(max(sh for sh in c["shades"] if sh is not None), 1.0)
        # lower latency is darker: the fastest series' cell has shade 1
        latency = board["cols"][7]
        best = min(range(6), key=lambda r: latency["values"][r])
        self.assertEqual(latency["shades"][best], 1.0)
        self.assertEqual(len(scatter["points"]), 6)
        self.assertTrue(all(x > 0 and 0 <= y <= 100 for _, x, y in scatter["points"]))
        # y fits the data with a margin, not 0 to 100
        y0, y1 = cb.scatter_domain(scatter)[2:]
        ys = [y for _, _, y in scatter["points"]]
        self.assertTrue(y0 < min(ys) and max(ys) < y1 and y1 - y0 <= max(ys) - min(ys) + 13)
        # without engine_peak_gib on every series, memory is the host peak and says so
        self.assertEqual(board["cols"][9]["head"], ("Memory", "host peak", "GiB ↓"))
        self.assertIn("host-wide", board["memory_note"])

    def test_engine_peak_memory(self):
        paths = self.field(4)
        labels = [f"Model {i} (sample)" for i in range(4)]
        meta = cb.parse_series_meta([f"{lab}:engine_peak_gib={g}" for lab, g in zip(labels, (86.0, 35.3, 0.58, 1.9))])
        series = cb.load_series(paths, {labels[0]: "First"}, meta)
        self.assertEqual(series[0].label, "First")
        self.assertEqual(series[0].data["meta"]["engine_peak_gib"], "86.0")
        mem = cb.scoreboard_panel(series)["cols"][-1]
        self.assertEqual(mem["head"], ("Memory", "engine peak", "GiB ↓"))
        self.assertEqual(mem["values"], [86.0, 35.3, 0.58, 1.9])
        self.assertEqual(mem["shades"][2], 1.0)  # lower is better
        # one series without it: back to the host peak
        partial = cb.load_series(paths, None, {k: v for k, v in meta.items() if k != labels[3]})
        self.assertEqual(cb.scoreboard_panel(partial)["cols"][-1]["head"][1], "host peak")
        # the CLI: summary.md names the source
        out = self.dir / "engine-peak"
        args = [a for lab, g in zip(labels, (86.0, 35.3, 0.58, 1.9)) for a in ("--series-meta", f"{lab}:engine_peak_gib={g}")]
        self.assertEqual(run_cli("report", *paths, "--out", out, "--no-png", "--no-zip", *args,
                                 "--relabel", f"{labels[0]}=First"), 0)
        summary = (out / "summary.md").read_text()
        self.assertIn("## Scoreboard", summary)
        self.assertIn("CapyCTL's measured peak of the engine process", summary)
        self.assertIn("| Memory engine peak GiB ↓ |", summary)
        self.assertRegex(summary, r"\| First \|.*\| 86\.0 \|")
        # files untouched; bad labels and values refused
        self.assertNotIn("engine_peak_gib", json.loads(paths[0].read_text())["meta"])
        with self.assertRaises(SystemExit) as cm:
            cb.load_series(paths, None, {"nobody": {"engine_peak_gib": "1"}})
        self.assertIn("--series-meta: no results file is labelled 'nobody'", str(cm.exception))
        with self.assertRaises(SystemExit):
            cb.parse_series_meta(["no-colon=1"])
        self.assertEqual(cb.parse_series_meta(["a:b:k=v=w"]), {"a:b": {"k": "v=w"}})
        bad = cb.load_series(paths, None, {lab: {"engine_peak_gib": "lots"} for lab in labels})
        with self.assertRaises(SystemExit):
            cb.scoreboard_panel(bad)

    def test_field_report(self):
        out = self.dir / "field"
        rc = run_cli("report", *self.field(6), "--out", out, "--summary", "--no-zip", "--title", "Six (sample)")
        self.assertEqual(rc, 0)
        page = (out / "summary" / "summary.html").read_text()
        self.assertIn('<table class="board">', page)
        for title in ("Scoreboard", "Quality vs speed"):
            self.assertIn(f"<h3>{title} <small>", page)
        self.assertNotIn("Latency p50, 1 question", page)
        self.assertNotIn('class="barpct"', page)
        self.assertIn('class="tick lbl"', page)
        # the scoreboard rows carry the chart's markers; no separate legend
        self.assertEqual(page.count('<svg class="mkr"'), 6)
        self.assertIn('<div class="legend s-legend"></div>', page)
        # values only: no change against the first series in summary.md or the report's tables
        summary = (out / "summary.md").read_text()
        self.assertIn("the tables give values only", summary)
        self.assertNotRegex(summary, r"\([+-]\d+\.\d (pt|%)\)")
        self.assertNotRegex((out / "report.html").read_text(), r"<small>[+-]\d+\.\d pt")
        names = sorted(p.name for p in (out / "summary").glob("*.png"))
        self.assertEqual(len(names), 4 if HAVE_MPL else 0)

    def test_relabel(self):
        out = self.dir / "relabel"
        rc = run_cli("report", *self.results, "--out", out, "--no-png", "--no-zip",
                     "--relabel", "Engine A (sample)=Alpha · fake", "--relabel", "Engine B (sample)=Beta")
        self.assertEqual(rc, 0)
        summary = (out / "summary.md").read_text()
        self.assertIn("| Point | Alpha · fake | Beta |", summary)
        # the results files stay as run wrote them
        self.assertEqual(self.data()["label"], "Engine A (sample)")
        self.assertEqual(cb.parse_relabel(["a=b=c"]), {"a": "b=c"})
        with self.assertRaises(SystemExit) as cm:
            cb.load_series(self.results, {"Engine C": "x"})
        self.assertIn("'Engine A (sample)', 'Engine B (sample)'", str(cm.exception))
        with self.assertRaises(SystemExit):
            cb.parse_relabel(["no-equals"])

    def test_latency_only_headline(self):
        # Without throughput or accuracy the summary leads with latency p50 bars (lower is better).
        data = self.data()
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

    def test_version_2_results_still_render(self):
        # A decisions run of schema 2: burst rounds, no cache test, AG News and BoolQ with accuracy and Brier.
        data = self.data()
        data["schema_version"] = 2
        dec = data["decisions"]
        dec.pop("manifest")
        dec.pop("cache")
        for p in dec["latency"]["points"]:
            for k in ("per_question_p50_s", "per_question_p95_s"):
                p["summary"].pop(k)
        cc = dec["concurrency"]
        cc["settings"] = {k: v for k, v in cc["settings"].items() if k in ("levels", "questions", "state_tokens")}
        cc["settings"]["rounds"] = 1
        for lv in cc["levels"]:
            sm = lv["summary"]
            v = sm.pop("requests_per_s")
            for k in ("questions_per_s", "error_rate", "wall_s"):
                sm.pop(k)
            sm["decisions_per_s"] = {"median": v, "min": v, "max": v, "n": 1}
            lv["rounds"] = [{"requests": lv.pop("requests"), "decisions_per_s": v}]
        old_sets = []
        for st in dec["accuracy"]["sets"]:
            if st.pop("kind") == "classic":
                st["summary"] = {k: st["summary"][k] for k in ("items", "answered", "errors", "accuracy", "brier",
                                                                "latency_s", "input_tokens", "memory_peak_gib")}
                old_sets.append(st)
        dec["accuracy"]["sets"] = old_sets
        old = self.dir / "v2.json"
        old.write_text(json.dumps(data))
        out = self.dir / "v2"
        self.assertEqual(run_cli("report", old, self.results[1], "--out", out, "--summary", "--no-png",
                                 "--no-zip"), 0)
        summary = (out / "summary.md").read_text()
        self.assertIn("### Requests per second (requests/s, higher is better)", summary)
        rps = cb.fmt(cb.level_get(cb.load_series([old])[0], 4, "requests_per_s"))
        self.assertIn(f"| 4 | {rps} |", summary)
        self.assertIn("burst rounds", (out / "report.html").read_text())

    def test_chat_and_decisions_are_not_mixed(self):
        chat = self.data()
        chat.pop("decisions")
        chat["mode"] = "chat"
        p = self.dir / "chat.json"
        p.write_text(json.dumps(chat))
        with self.assertRaises(SystemExit) as cm:
            cb.load_series([self.results[0], p])
        self.assertIn("report them separately", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
