"""Workspace wrapper. Delegates to realtime_bangla_taka_detection/run_research_pipeline.py."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "realtime_bangla_taka_detection" / "run_research_pipeline.py"
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
