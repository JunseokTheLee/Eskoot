import cv2
import torch
import torchvision
import numpy as np
from torchvision.transforms import v2 as T

# ---------------- CONFIG ----------------
WEIGHTS = "ssdlite_mobilenetv3_roadhazards.pth"
IMG_SIZE = 320
CONF_THRES = 0.25
VIDEO_PATH = "./01_10072023.mp4"   # or 0 for webcam
# ----------------------------------------

CLASS_NAMES = ["pothole", "crack", "manhole"]
NUM_CLASSES = len(CLASS_NAMES) + 1  # background

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights=None,
    num_classes=NUM_CLASSES
)

ckpt = torch.load(WEIGHTS, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()

# ---------------- PREPROCESS ----------------
transform = T.Compose([
    T.ToImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToDtype(torch.float32, scale=True),
])

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    slowdown = 2.0   # 2.0 = 2× slower, 3.0 = 3× slower
    delay = int((1000 / fps) * slowdown)
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess
    img = transform(rgb).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(img)[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    # Scale boxes back to original frame size
    sx, sy = w / IMG_SIZE, h / IMG_SIZE

    for box, score, label in zip(boxes, scores, labels):
        if score < CONF_THRES:
            continue

        x1, y1, x2, y2 = box
        x1, x2 = int(x1 * sx), int(x2 * sx)
        y1, y2 = int(y1 * sy), int(y2 * sy)

        cls = label - 1  # remove background offset
        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(label)

        color = (0, 0, 255) if name in ["pothole", "crack", "manhole"] else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Road Hazard Detection (MobileNet)", frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
