"""Repo-root paths. Data stays at the workspace root; code lives under flyfx/."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

CONNECTOME_DIR = PROJECT_ROOT / "fly-connectome"
BRAIN_DIR = PROJECT_ROOT / "brains"
REPORTS_DIR = PROJECT_ROOT / "reports"
SNAPSHOTS_DIR = PROJECT_ROOT / "snapshots"
MT4_DIR = PROJECT_ROOT / "mt4"
VENDOR_DIR = PROJECT_ROOT / "vendor"
BANC_PROJECT = VENDOR_DIR / "BANC-project-main"
BANC_CACHE = PROJECT_ROOT / "fly-banc"
CONFIG_DIR = PROJECT_ROOT / "configs"
DASH_HTML = PACKAGE_DIR / "ui" / "web" / "dash.html"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
