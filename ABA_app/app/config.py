#### PATCHARAKORN ####

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = APP_ROOT.parent.parent  # d:\user\ReaLearn

# Task 3 gold data lives in the ABA_mining pipeline project, not copied here — read directly
# from there so this app always reflects the same data the pipeline produces.
ABA_MINING_ROOT = WORKSPACE_ROOT / "internship" / "ABA_mining"
TASK3_GOLD_DIR = ABA_MINING_ROOT / "Dataset" / "Task3"

FRONTEND_DIR = APP_ROOT / "frontend"

# Aspect -> gold CSV filename (the single "Contrary(P)Body(N)" sheet each has, per
# Task3_Implementation_Report.md). Only aspects with a real human-voted gold file are listed;
# see ABA_mining's src/task3.py for the full multi-category loader this is a simplified
# read of (this app only needs the Contrary(P)Body(N) category for now).
ASPECT_GOLD_FILES = {
    "check-in": "2. Verify - Task 3 - Check-in (Silver).xlsx - Contrary(P)Body(N).csv",
    "check-out": "1. Verify - Task 3 - Check-out (Silver).xlsx - Contrary(P)Body(N).csv",
    "price": "3. Verify - Task 3 - Price (Silver).xlsx - Contrary(P)Body(N).csv",
    "staff": "4. Verify - Task 3 - Staff (Silver).xlsx - Contrary(P)Body(N).csv",
}
