"""Live lab dashboard: three MaleCNS heads, 2oo3 committee, snowball, P&L.

Stdlib HTTP server. Open http://127.0.0.1:8765 while the trader runs with --gui.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "web" / "dash.html"


def compact_activity(brain, x: np.ndarray) -> dict:
    """Downsample one head's 166k state into something a browser can paint."""
    retina = getattr(brain, "retina", np.array([], dtype=np.int32))
    desc = getattr(brain, "descending", np.array([], dtype=np.int32))
    ret = x[retina] if retina.size else x[:128]
    mid = max(ret.size // 2, 1)
    motor = x[desc] if desc.size else x[:64]

    def bins(arr: np.ndarray, n: int) -> list[float]:
        if arr.size == 0:
            return [0.0] * n
        idx = np.linspace(0, arr.size - 1, n).astype(np.int32)
        return np.abs(arr[idx]).clip(0.0, 1.0).round(3).tolist()

    return {
        "eye_bull": bins(ret[:mid], 64),
        "eye_bear": bins(ret[mid:], 64),
        "motor": bins(motor, 48),
        "retina_bull": round(float(np.mean(np.abs(ret[:mid]))), 3),
        "retina_bear": round(float(np.mean(np.abs(ret[mid:]))), 3),
    }


def pack_head(
    brain,
    x: np.ndarray,
    vote: str,
    score: float,
    nfire: int,
    tau: float,
    role: str,
    plastic=None,
) -> dict:
    out = compact_activity(brain, x)
    tau = float(max(tau, 1e-6))
    out.update(
        {
            "role": role,
            "vote": vote,
            "score": round(float(score), 2),
            "tau": round(tau, 2),
            "nfire": int(nfire),
            "norm": round(float(score) / tau, 3),
        }
    )
    if plastic is not None:
        out["gain_pos"] = round(float(plastic.gain_pos), 3)
        out["gain_neg"] = round(float(plastic.gain_neg), 3)
        out["da"] = round(float(plastic.last_r), 3)
        out["n_updates"] = int(plastic.n_updates)
    return out


class DashHub:
    def __init__(self, port: int = 8765) -> None:
        self.port = int(port)
        self.lock = threading.Lock()
        self.state: dict = {"ok": True, "live": False}
        self.log: deque[dict] = deque(maxlen=36)
        self.history: dict[str, deque] = {
            "price": deque(maxlen=120),
            "equity": deque(maxlen=120),
            "trend": deque(maxlen=120),
            "fade": deque(maxlen=120),
            "conf": deque(maxlen=120),
        }
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def publish(self, payload: dict, log_line: dict | None = None) -> None:
        with self.lock:
            if payload:
                self.state = payload
                self.state["ok"] = True
                self.state["live"] = True
            for key, buf in self.history.items():
                val = payload.get(key)
                if val is None and key in ("trend", "fade", "conf"):
                    head = (payload.get("heads") or {}).get(key) or {}
                    val = head.get("score")
                if val is None and key == "price":
                    val = payload.get("close")
                if val is None and key == "equity":
                    val = (payload.get("account") or {}).get("equity")
                if val is not None:
                    try:
                        buf.append(round(float(val), 5))
                    except (TypeError, ValueError):
                        pass
            self.state["history"] = {k: list(v) for k, v in self.history.items()}
            self.state["log"] = list(self.log)
            if log_line:
                self.log.append(log_line)
                self.state["log"] = list(self.log)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self, open_browser: bool = True) -> str:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/api/state"):
                    blob = json.dumps(hub.snapshot()).encode("utf-8")
                    self._send(200, blob, "application/json; charset=utf-8")
                    return
                html = HTML_PATH.read_text(encoding="utf-8")
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        last_err = None
        for port in range(self.port, self.port + 8):
            try:
                self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                self.port = port
                break
            except OSError as exc:
                last_err = exc
                self._httpd = None
        if self._httpd is None:
            raise RuntimeError(f"could not bind dashboard port: {last_err}")
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://127.0.0.1:{self.port}/"
        print(f"dashboard  {url}  (three fly heads + BANC risk, 2oo3, dopamine, snowball)")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
