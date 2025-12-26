import cv2
import torch
import numpy as np
import time
from collections import deque
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ---------------- CONFIG ----------------asdf
WEIGHTS = "runs/train/roadhazards_y5n3/weights/best.pt"
IMG_SIZE = 640
CONF_THRES = 0.15
IOU_THRES = 0.45
CAMERA_INDEX = 0
# ----------------------------------------
fps_buffer = deque(maxlen=30)
# Device (CPU / MPS / CUDA)
device = select_device("cpu")  # "cpu", "mps", or "0"
VIDEO_PATH = "./01_10072023.mp4"
# Load model
model = DetectMultiBackend(WEIGHTS, device=device)
stride = model.stride
names = model.names

# FP16 only on CUDA
fp16 = device.type == "cuda"
model.model.half() if fp16 else None

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    slowdown = 2.0   # 2.0 = 2× slower, 3.0 = 3× slower
    delay = int((1000 / fps) * slowdown)
    ret, frame = cap.read()
    start_time = time.time()
    if not ret:
        break
        
    # Letterbox resize
    img = letterbox(frame, IMG_SIZE, stride=stride, auto=True)[0]

    # Convert BGR → RGB → tensor
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if fp16 else img.float()
    img /= 255.0

    if img.ndim == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)

    # NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)

    # Draw boxes
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls)]

                color = (0, 0, 255) if label in ["pothole", "crack", "manhole"] else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
    end_time = time.time()
    frame_time = end_time - start_time
    fps_buffer.append(1.0 / frame_time)
    fps = sum(fps_buffer) / len(fps_buffer)
    cv2.putText(
    frame,
    f"FPS: {fps:.2f}",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2
)
    cv2.imshow("Road Hazard Detection", frame)
    
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
