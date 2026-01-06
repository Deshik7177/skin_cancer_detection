from ultralytics import YOLO
from pathlib import Path
import cv2
import uuid

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best.pt"
OUTPUT_DIR = BASE_DIR.parent / "static" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))

def run_inference(image_path: Path):
    results = model.predict(source=str(image_path), conf=0.25, verbose=False)

    img = cv2.imread(str(image_path))
    detections = []

    for r in results:
        # 🔹 Detection model
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                label = model.names[cls]
                detections.append({"label": label, "confidence": round(conf * 100, 2)})

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 180), 2)
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 180),
                    2,
                )

        # 🔹 Classification model
        elif hasattr(r, "probs") and r.probs is not None:
            cls = int(r.probs.top1)
            conf = float(r.probs.top1conf)
            label = model.names[cls]
            detections.append({"label": label, "confidence": round(conf * 100, 2)})

    output_name = f"{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(OUTPUT_DIR / output_name), img)

    return output_name, detections
