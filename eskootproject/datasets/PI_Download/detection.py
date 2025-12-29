import cv2
import time
import torch
import torchvision
from collections import deque

# ---------------- CONFIG ----------------
def leftsignal():
    print("left signal sent!")
def rightsignal():
    print("right signal sent!")
def frontsignal():
    print("front signal sent!")
ROI_Y_START_FRAC =0.5
ROY_Y_END_FRAC = 1
LANE_LEFT_MAX  = 0.33
LANE_RIGHT_MIN = 0.66
SIGNAL_COOLDOWN = 1.5
last_signal_time = 0.0

WEIGHTS = "ssdlite_mobilenetv3_roadhazards.pth"
VIDEO_PATH = "./01_10072023.mp4" 
IMG_SIZE = 320                  
CONF_THRES = 0.40


DETECT_EVERY = 2  

CLASS_NAMES = ["pothole", "crack", "manhole"]
NUM_CLASSES = len(CLASS_NAMES) + 1  
DEVICE = torch.device("cpu")


torch.set_num_threads(2)
torch.set_num_interop_threads(1)
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(1)
except:
    pass

# ---------------- LOAD MODEL ----------------
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
    weights=None,
    num_classes=NUM_CLASSES
)

ckpt = torch.load(WEIGHTS, map_location="cpu")
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model.eval()

# ---------------- TRACE WRAPPER (simple JIT) ----------------
class TraceWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor):
        det = self.m([x])[0]
        return det["boxes"], det["scores"], det["labels"]

wrapped = TraceWrapper(model).eval()

example = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
net = torch.jit.trace(wrapped, example, strict=False)
try:
    net = torch.jit.optimize_for_inference(net)
except Exception:
    pass
net.eval()

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
slowdown = 1.0
delay = int((1000 / fps_src) * slowdown)

fps_buffer = deque(maxlen=30)


cached_boxes = None
cached_scores = None
cached_labels = None
cached_status = "none"
cached_best = None  # (x1,y1,x2,y2,name,score)

frame_idx = 0

# ---------------- LOOP ----------------
with torch.inference_mode():
    frame_count = 0
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        h, w = frame.shape[:2]

        frame_idx += 1
        run_det = (frame_idx % DETECT_EVERY == 0) or (cached_scores is None)

        if run_det:
            # resize -> RGB

            y0 = int(h*ROI_Y_START_FRAC)
            y1 = int(h*ROY_Y_END_FRAC)
            roi = frame[y0:y1, :]
            h_roi, w_roi = roi.shape[:2]
            resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # HWC uint8 -> CHW float32 [0,1]
            x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
            x = x.to(dtype=torch.float32).mul_(1.0 / 255.0)

            boxes, scores, labels = net(x)

            cached_boxes, cached_scores, cached_labels = boxes, scores, labels
            cached_status = "none"
            cached_best = None

            if scores.numel() > 0:
                best_i = int(torch.argmax(scores).item())
                best_score = float(scores[best_i].item())

                if best_score >= CONF_THRES:
                    cached_status = "present"

                    box = boxes[best_i]
                    label = int(labels[best_i].item())

                    # scale box back to original size
                    sx, sy = w_roi / IMG_SIZE, h_roi / IMG_SIZE
                    x1 = int(box[0].item() * sx)
                    yy1 = int(box[1].item() * sy)
                    x2 = int(box[2].item() * sx)
                    yy2 = int(box[3].item() * sy)
                    y1_full = yy1 + y0
                    y2_full = yy2 + y0
                    cls = label - 1
                    name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(label)

                    
                    cx = 0.5 * (x1 + x2)          # center in full-frame x coords
                    cx_norm = cx / max(w, 1)      # 0..1
                    cached_best = (x1, y1_full, x2, y2_full, name, best_score, cx_norm)


        
        if cached_best is not None:
            x1, y1, x2, y2, name, best_score, cx_norm = cached_best
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} {best_score:.2f}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        
        now = time.time()
        if cached_status != "none" and (now - last_signal_time >= SIGNAL_COOLDOWN):
            _,_, _, _, _, _, cx_norm = cached_best
            if (cx_norm< LANE_LEFT_MAX):
                leftsignal()
            elif (cx_norm>LANE_RIGHT_MIN):
                rightsignal()
            else:
                frontsignal()
            
            last_signal_time = now

   
        dt = time.time() - t0
        fps_buffer.append(1.0 / max(dt, 1e-6))
        fps_avg = sum(fps_buffer) / len(fps_buffer)

        cv2.putText(frame, f"FPS: {fps_avg:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {cached_status}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Detect every: {DETECT_EVERY} frames", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Road Hazard Detection (trace + frame-skip)", frame)
        
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
