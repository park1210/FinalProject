import argparse
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from api_client import GameAPIClient, GameAPIError
from difference_analyzer import DifferenceAnalyzer
from image_detector import ImageDetector

LOGGER = logging.getLogger("game_worker")


def process_job(api, detector, job):
    game_id = job["id"]
    with TemporaryDirectory(prefix="game_worker_") as temporary:
        root = Path(temporary)
        before_path = api.download(job["before_download"], root / "before.jpg")
        after_path = api.download(job["after_download"], root / "after.jpg")
        before_counts = detector.detect(before_path)
        after_counts = detector.detect(after_path)
        difference = DifferenceAnalyzer.compare(before_counts, after_counts)
        api.complete(game_id, before_counts, after_counts, difference)
        LOGGER.info("Completed %s: %s", game_id, difference)


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 game analysis worker")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA device")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Process one job or exit when queue empty")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api = GameAPIClient()
    detector = ImageDetector(device=args.device)  # Load pretrained model once
    LOGGER.info("Worker started")

    while True:
        try:
            job = api.claim()
        except (GameAPIError, ValueError) as exc:
            LOGGER.error("Claim failed: %s", exc)
            if args.once:
                raise
            time.sleep(max(args.interval, 1))
            continue

        if job is None:
            if args.once:
                LOGGER.info("No READY games")
                break
            time.sleep(max(args.interval, 1))
            continue

        try:
            process_job(api, detector, job)
        except Exception:
            LOGGER.exception("Analysis failed for game %s", job["id"])
            try:
                api.fail(job["id"], "Edge inference failed")
            except GameAPIError:
                LOGGER.exception("Could not report failure; manual recovery may be required")
            if args.once:
                raise

        if args.once:
            break


if __name__ == "__main__":
    main()
