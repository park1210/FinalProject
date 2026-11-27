
from collections import Counter


class DifferenceAnalyzer:
    """Compare object counts in two images."""

    @staticmethod
    def compare(before_counts, after_counts):
        if not isinstance(before_counts, dict):
            raise TypeError("before_counts must be a dict")

        if not isinstance(after_counts, dict):
            raise TypeError("after_counts must be a dict")

        for counts in (before_counts, after_counts):
            for name, count in counts.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Invalid object name")

                if type(count) is not int or count < 0:
                    raise ValueError(
                        "Object counts must be non-negative integers"
                    )

        before = Counter(before_counts)
        after = Counter(after_counts)

        added = after - before
        removed = before - after

        return {
            "added": dict(added),
            "removed": dict(removed),
        }


def compare_object_counts(before_counts, after_counts):
    return DifferenceAnalyzer.compare(
        before_counts,
        after_counts,
    )


if __name__ == "__main__":
    before = {
        "person": 1,
        "cell phone": 1,
        "bottle": 2,
    }

    after = {
        "person": 1,
        "bottle": 1,
        "book": 1,
    }

    result = compare_object_counts(before, after)

    print("BEFORE:", before)
    print("AFTER :", after)
    print("RESULT:", result)
