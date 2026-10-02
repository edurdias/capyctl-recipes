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
        self.assertIn("Generation, one stream", page)
        self.assertIn("Throughput vs streams", page)
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

    @unittest.skipIf(HAVE_MPL, "matplotlib is installed")
    def test_report_without_matplotlib_says_so(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cb.main(["report", str(self.results[0]), "--out", str(self.dir / "nompl")])
        self.assertIn("matplotlib is not installed", err.getvalue())
        self.assertIn("uv run --with matplotlib", err.getvalue())


if __name__ == "__main__":
    unittest.main()
