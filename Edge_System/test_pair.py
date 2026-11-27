
import argparse
import json
from pathlib import Path

from image_detector import ImageDetector
from difference_analyzer import DifferenceAnalyzer


def analyze_pair(before_path, after_path, device="cpu"):
    before_path = Path(before_path)
    after_path = Path(after_path)

    if not before_path.is_file():
        raise FileNotFoundError(
            f"Before image not found: {before_path}"
        )

    if not after_path.is_file():
        raise FileNotFoundError(
            f"After image not found: {after_path}"
        )

    # Load the pretrained model once
    detector = ImageDetector(device=device)

    # Detect objects in each image
    before_counts = detector.detect(before_path)
    after_counts = detector.detect(after_path)

    # Compare class counts
    difference = DifferenceAnalyzer.compare(
        before_counts,
        after_counts,
    )

    return {
        "before": before_counts,
        "after": after_counts,
        "difference": difference,
    }


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv5 Before/After image comparison"
    )

    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    result = analyze_pair(
        before_path=args.before,
        after_path=args.after,
        device=args.device,
    )

    print("\n=== YOLOv5 PAIRWISE RESULT ===")
    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
