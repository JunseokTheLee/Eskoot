

from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import v2 as T

# =============================
# CONFIG
# =============================
CLASS_NAMES = ["pothole", "crack", "manhole"]
NUM_CLASSES = len(CLASS_NAMES) + 1  # background

BATCH_SIZE = 8
EPOCHS = 40
LR = 0.002
TRAIN_IMG_DIR = "C:/Users/leeja/Documents/GitHub/Eskoot/eskootproject/datasets/roadhazards/images/train"
TRAIN_LBL_DIR = "C:/Users/leeja/Documents/GitHub/Eskoot/eskootproject/datasets/roadhazards/labels/train"
VAL_IMG_DIR   = "C:/Users/leeja/Documents/GitHub/Eskoot/eskootproject/datasets/roadhazards/images/val"
VAL_LBL_DIR   = "C:/Users/leeja/Documents/GitHub/Eskoot/eskootproject/datasets/roadhazards/labels/val"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# DATASET (YOLO bbox labels)
# =============================
class YoloBBoxDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, transforms=None):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.transforms = transforms
        self.images = sorted([p for p in self.img_dir.iterdir()
                              if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cls, xc, yc, bw, bh = map(float, line.split())
                cls = int(cls)

                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h

                # skip degenerate
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # background = 0

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    print("Device:", DEVICE)
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    tf = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    train_ds = YoloBBoxDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, transforms=tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # ✅ IMPORTANT:
    # weights="DEFAULT" forces COCO classes (91) -> your error.
    # So we use pretrained BACKBONE only:
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone="DEFAULT",
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0

        for images, targets in train_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total/len(train_loader):.4f}")

    torch.save({"model": model.state_dict(), "classes": CLASS_NAMES},
               "ssdlite_mobilenetv3_roadhazards.pth")
    print("Saved: ssdlite_mobilenetv3_roadhazards.pth")


if __name__ == "__main__":
    main()
