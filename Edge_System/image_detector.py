
from collections import Counter

import torch


class ImageDetector:
    def __init__(self, device="cpu"):
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5s",
            pretrained=True,
            trust_repo=True
        )

        self.model.to(device)
        self.model.eval()

    def detect(self, image_path):
        results = self.model(str(image_path))

        detections = results.pandas().xyxy[0]

        counts = Counter(
            detections["name"].tolist()
        )

        return dict(counts)


if __name__ == "__main__":
    detector = ImageDetector()

    print(
        detector.detect("test_images/before.jpg")
    )
