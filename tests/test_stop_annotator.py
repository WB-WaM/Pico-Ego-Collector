"""Real socket/process regressions for restarting the annotation server."""
import importlib.util
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stop_annotator.py"
spec = importlib.util.spec_from_file_location("stop_annotator", SCRIPT)
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)

LISTENER = """
import argparse, signal
from http.server import HTTPServer, BaseHTTPRequestHandler
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int)
parser.add_argument('--ignore-term', action='store_true')
args = parser.parse_args()
if args.ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ready')
HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
"""


@unittest.skipUnless(sys.platform == "linux", "Linux /proc process management")
class RestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.processes = []
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]

    def tearDown(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)
        self.temp.cleanup()

    def request(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=.2) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"ready")

    def launch(self, root=None, ignore_term=False):
        root = root or self.root
        script = root / "scripts/serve_annotator.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(LISTENER)
        args = [sys.executable, str(script), "--port", str(self.port)]
        if ignore_term:
            args.append("--ignore-term")
        proc = subprocess.Popen(args, cwd=root, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(proc)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.assertIsNone(proc.poll())
            try:
                self.request()
                return proc
            except OSError:
                time.sleep(.02)
        self.fail("Listener did not start")

    def check_restart(self, paused=False, ignore_term=False):
        proc = self.launch(ignore_term=ignore_term)
        if paused:
            os.kill(proc.pid, signal.SIGSTOP)
            deadline = time.monotonic() + 2
            while stop.process_identity(proc.pid)[0] != "T" and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(stop.process_identity(proc.pid)[0], "T")
        stop.stop_previous(self.root, "127.0.0.1", self.port, timeout=.25)
        self.assertIsNotNone(proc.poll())
        replacement = self.launch()
        self.assertNotEqual(replacement.pid, proc.pid)
        self.request()

    def test_running_server(self):
        self.check_restart()

    def test_stopped_server_and_unreaped_child(self):
        self.check_restart(paused=True)

    def test_unresponsive_server_escalates(self):
        self.check_restart(ignore_term=True)

    def test_other_project_listener_is_preserved(self):
        proc = self.launch(root=self.root / "other_project")
        with self.assertRaisesRegex(RuntimeError, "PICO_WEB_PORT"):
            stop.stop_previous(self.root, "127.0.0.1", self.port, timeout=.25)
        self.assertIsNone(proc.poll())
        self.request()


if __name__ == "__main__":
    unittest.main()
