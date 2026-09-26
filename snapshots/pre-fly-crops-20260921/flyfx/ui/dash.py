"""Live lab dashboard: category committee, MaleCNS heads, snowball, P&L.

Stdlib HTTP server. Open http://127.0.0.1:8765 while the trader runs with --gui.
POST /api/command to pick a pair and start a fast replay without MT4.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from flyfx.brain.fly_brains import list_brain_symbols, list_evo_policies
from flyfx.paths import DASH_HTML as HTML_PATH
from flyfx.sense.category_kalman import CATEGORY_NAMES, parse_fuse_cats
from flyfx.brain.sugar_feed import IDLE as SUGAR_IDLE
from flyfx.sense.fuse_regime import idle_fuse_plan


def idle_cat_fuse() -> dict:
    return {
        "impulse": 0.0,
        "fused": 0.0,
        "residual": 0.0,
        "available": False,
        "n_live": 0,
        "n_expected": len(CATEGORY_NAMES),
        "regime": "CHOP",
        "agreement": 0.0,
        "uncertainty": 1.0,
        "parts": {},
        "names": [],
        "selected": list(CATEGORY_NAMES),
    }


def pack_category_sensors(categories: dict | None, mix: dict | None = None) -> dict:
    """7 Kalman rows without fly votes — used by the cat-fuse 3-voter GUI."""
    idle = idle_categories()
    rows = dict(idle["categories"])
    for name, st in (categories or {}).items():
        base = dict(rows.get(name) or {})
        base.update(
            {
                "available": bool(st.get("available")),
                "impulse": st.get("impulse", 0.0),
                "fused": st.get("fused", 0.0),
                "residual": st.get("residual", 0.0),
                "agreement": st.get("agreement", 0.0),
                "n_live": st.get("n_live", 0),
                "n_expected": st.get("n_expected", 0),
                "regime": st.get("regime") or "",
                "channels": list(st.get("channels") or []),
                "parts": dict(st.get("parts") or {}),
            }
        )
        rows[name] = base
    fuse = mix or idle_cat_fuse()
    idle["categories"] = rows
    idle["n_live"] = fuse.get("n_live", 0)
    idle["agreement"] = fuse.get("agreement", 0.0)
    idle["impulse"] = fuse.get("impulse", 0.0)
    idle["fused"] = fuse.get("fused", 0.0)
    idle["regime"] = fuse.get("regime") or ""
    return idle


def idle_categories() -> dict:
    """Empty 7-row committee so the GUI shows the pipeline before the first bar."""
    rows = {
        name: {
            "decision": "HOLD",
            "vote_type": "HOLD",
            "conf_norm": 0.0,
            "trend": "HOLD",
            "fade": "HOLD",
            "tech": "HOLD",
            "tech_reason": "",
            "inhibited": False,
            "available": False,
            "impulse": 0.0,
            "agreement": 0.0,
            "n_live": 0,
            "n_expected": 0,
            "regime": "",
            "n_agree": 0,
            "channels": [],
            "parts": {},
        }
        for name in CATEGORY_NAMES
    }
    return {
        "winner": "",
        "vote_type": "HOLD",
        "conf": 0.0,
        "inhibited": [],
        "losers": [],
        "conflict": "",
        "categories": rows,
    }


def idle_nodes() -> dict:
    """Empty 7×4 node board (tech + fly-trend + fly-fade + fly-conf)."""
    empty_head = {
        "vote": "HOLD",
        "score": 0.0,
        "tau": 1.0,
        "nfire": 0,
        "norm": 0.0,
        "eye_bull": [],
        "eye_bear": [],
        "motor": [],
        "compact": True,
    }
    board: dict = {}
    for name in CATEGORY_NAMES:
        board[name] = {
            "tech": {"vote": "HOLD", "reason": "", "strength": 0.0},
            "heads": {
                "trend": {**empty_head, "role": f"{name}-trend"},
                "fade": {**empty_head, "role": f"{name}-fade"},
                "conf": {**empty_head, "role": f"{name}-conf"},
            },
            "decision": "HOLD",
            "vote_type": "HOLD",
            "inhibited": False,
            "available": False,
            "winner": False,
            "impulse": 0.0,
            "conf_norm": 0.0,
            "n_agree": 0,
        }
    return board


def sensor_nodes(categories: dict | None, include: tuple[str, ...] | list[str] | None = None) -> dict:
    """Kalman-only node board (no 21 fly heads) for the fused 3-voter path."""
    board = idle_nodes()
    try:
        selected = set(parse_fuse_cats(include))
    except ValueError:
        selected = set(CATEGORY_NAMES)
    for name, st in (categories or {}).items():
        row = dict(board.get(name) or {})
        row.update(
            {
                "available": bool(st.get("available")),
                "impulse": st.get("impulse", 0.0),
                "fused": st.get("fused", 0.0),
                "residual": st.get("residual", 0.0),
                "agreement": st.get("agreement", 0.0),
                "n_live": st.get("n_live", 0),
                "n_expected": st.get("n_expected", 0),
                "regime": st.get("regime") or "",
                "in_fuse": name in selected,
                "channels": list(st.get("channels") or []),
                "parts": dict(st.get("parts") or {}),
                "heads": {},
            }
        )
        board[name] = row
    return board


class DashServer(ThreadingHTTPServer):
    """Refuse SO_REUSEADDR so a zombie --gui cannot share :8765 and steal the browser."""

    allow_reuse_address = False
    daemon_threads = True


def compact_activity(brain, x: np.ndarray, eye_n: int = 64, motor_n: int = 48) -> dict:
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
        "eye_bull": bins(ret[:mid], int(eye_n)),
        "eye_bear": bins(ret[mid:], int(eye_n)),
        "motor": bins(motor, int(motor_n)),
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
    viz: bool = True,
    compact: bool = False,
) -> dict:
    eye_n = 16 if compact else 64
    motor_n = 24 if compact else 48
    out = compact_activity(brain, x, eye_n, motor_n) if viz else {
        "eye_bull": [],
        "eye_bear": [],
        "motor": [],
        "retina_bull": 0.0,
        "retina_bear": 0.0,
    }
    tau = float(max(tau, 1e-6))
    out.update(
        {
            "role": role,
            "vote": vote,
            "score": round(float(score), 2),
            "tau": round(tau, 2),
            "nfire": int(nfire),
            "norm": round(float(score) / tau, 3),
            "compact": bool(compact),
        }
    )
    if plastic is not None:
        out["gain_pos"] = round(float(plastic.gain_pos), 3)
        out["gain_neg"] = round(float(plastic.gain_neg), 3)
        out["da"] = round(float(plastic.last_r), 3)
        out["n_updates"] = int(plastic.n_updates)
    return out


class DashHub:
    def __init__(self, port: int = 8765, pairs: list[str] | None = None) -> None:
        self.port = int(port)
        self.lock = threading.Lock()
        self.state: dict = {
            "ok": True,
            "live": False,
            "running": False,
            "mode": "idle",
            "committee_mode": "fuse",
            "categories": idle_categories(),
            "cat_fuse": idle_cat_fuse(),
            "fuse_plan": idle_fuse_plan(),
            "decision": "HOLD",
            "tag": "",
            "balance": 100000.0,
            "nodes": idle_nodes(),
            "note": (
                "Seven category Kalmans inverse-variance mix into one impulse. "
                "One tech + fly-trend + fly-fade vote 2oo3. Fly-conf may only abstain."
            ),
            "risk_tol": "balanced",
            "sugar": dict(SUGAR_IDLE),
        }
        self.log: deque[dict] = deque(maxlen=36)
        self.history: dict[str, deque] = {
            "price": deque(maxlen=120),
            "equity": deque(maxlen=120),
            "trend": deque(maxlen=120),
            "fade": deque(maxlen=120),
            "conf": deque(maxlen=120),
        }
        self.commands: queue.Queue = queue.Queue(maxsize=8)
        self.cancel = threading.Event()
        self.pairs = list(pairs or [])
        self.cns_layout: dict = {"n": 0}
        self._last_pub = 0.0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def want_publish(self, force: bool = False, every_ms: float = 80.0) -> bool:
        now = time.perf_counter()
        if force or (now - self._last_pub) * 1000.0 >= every_ms:
            self._last_pub = now
            return True
        return False

    def publish(self, payload: dict, log_line: dict | None = None) -> None:
        with self.lock:
            if payload:
                self.state.update(payload)
                self.state["ok"] = True
                self.state["live"] = True
            for key, buf in self.history.items():
                val = payload.get(key) if payload else None
                if val is None and key in ("trend", "fade", "conf"):
                    head = ((payload or {}).get("heads") or {}).get(key) or {}
                    val = head.get("score")
                if val is None and key == "price":
                    val = (payload or {}).get("close")
                if val is None and key == "equity":
                    val = ((payload or {}).get("account") or {}).get("equity")
                if val is not None:
                    try:
                        buf.append(round(float(val), 5))
                    except (TypeError, ValueError):
                        pass
            self.state["history"] = {k: list(v) for k, v in self.history.items()}
            if log_line:
                self.log.append(log_line)
            self.state["log"] = list(self.log)

    def set_cns_layout(self, layout: dict) -> None:
        with self.lock:
            self.cns_layout = dict(layout or {"n": 0})

    def reset_run(self, symbol: str, mode: str) -> None:
        with self.lock:
            self.log.clear()
            for buf in self.history.values():
                buf.clear()
            self.state = {
                "ok": True,
                "live": True,
                "running": True,
                "mode": mode,
                "symbol": symbol,
                "label": f"{mode} {symbol}",
                "committee_mode": "fuse",
                "categories": idle_categories(),
                "cat_fuse": idle_cat_fuse(),
                "fuse_plan": idle_fuse_plan(),
                "nodes": idle_nodes(),
                "decision": "HOLD",
                "tag": "",
                "log": [],
                "sugar": dict(SUGAR_IDLE),
                "history": {k: [] for k in self.history},
            }
        self._last_pub = 0.0

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def meta(self) -> dict:
        with self.lock:
            return {
                "ok": True,
                "pairs": self.pairs,
                "brains": list_brain_symbols(),
                "evo": list_evo_policies(),
                "symbol": self.state.get("symbol") or "",
                "mode": self.state.get("mode") or "idle",
                "running": bool(self.state.get("running")),
                "port": self.port,
                "balance": float(self.state.get("balance") or 100000),
                "timeframes": ["NATIVE", "M1", "M5", "M10", "M15", "M30", "H1", "H4", "D1"],
            }

    def start(self, open_browser: bool = True) -> str:
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self._cors()
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/api/cns-layout":
                    blob = json.dumps(hub.cns_layout or {"n": 0}, separators=(",", ":")).encode("utf-8")
                    self._send(200, blob, "application/json; charset=utf-8")
                    return
                if path == "/api/state":
                    blob = json.dumps(hub.snapshot(), separators=(",", ":")).encode("utf-8")
                    self._send(200, blob, "application/json; charset=utf-8")
                    return
                if path == "/api/meta":
                    blob = json.dumps(hub.meta()).encode("utf-8")
                    self._send(200, blob, "application/json; charset=utf-8")
                    return
                if path in ("/", "/index.html", "/dash.html"):
                    html = HTML_PATH.read_text(encoding="utf-8")
                    self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                    return
                self._send(404, b"not found", "text/plain")

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path != "/api/command":
                    self._send(404, b"not found", "text/plain")
                    return
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(n) if n else b"{}"
                    cmd = json.loads(raw.decode("utf-8") or "{}")
                    if not isinstance(cmd, dict):
                        raise ValueError("command must be an object")
                except Exception as exc:
                    self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(), "application/json")
                    return
                action = str(cmd.get("action") or "")
                if action in ("stop", "replay", "live", "train"):
                    hub.cancel.set()
                try:
                    hub.commands.put_nowait(cmd)
                except queue.Full:
                    try:
                        hub.commands.get_nowait()
                    except queue.Empty:
                        pass
                    hub.commands.put_nowait(cmd)
                self._send(200, b'{"ok":true}', "application/json; charset=utf-8")

        requested = int(self.port)
        last_err = None
        for port in range(requested, requested + 8):
            try:
                self._httpd = DashServer(("127.0.0.1", port), Handler)
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
        print(f"dashboard  {url}  pick a pair, Replay, or Train DA (fast sim, no MT4 required)")
        if self.port != requested:
            print(f"dashboard  port {requested} was busy (another fly_forex.py --gui?). using {self.port}")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return url

    def stop(self) -> None:
        self.cancel.set()
        if self._httpd is not None:
            self._httpd.shutdown()
