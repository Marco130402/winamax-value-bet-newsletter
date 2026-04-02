"""
Long-running scheduler — fires the pipeline every Friday at 08:00 local time.

Usage:
    python scheduler.py

Run in background (macOS/Linux):
    nohup python scheduler.py > logs/scheduler.log 2>&1 &
"""

import logging
import os
import time

import schedule
from dotenv import load_dotenv

from main import run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)


def _job() -> None:
    try:
        run_pipeline()
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)


schedule.every().friday.at("08:00").do(_job)
log.info("Scheduler started — pipeline will run every Friday at 08:00.")

while True:
    schedule.run_pending()
    time.sleep(60)
