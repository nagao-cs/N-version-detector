from ultralytics import YOLO

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class YOLOv8Detector:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')  # 事前にダウンロード必要

    def predict(self, image_path):
        results = self.model(image_path)
        result = results[0]  # YOLOv8は1画像でも list[Results] が返る
        boxes = result.boxes

        # 各Box情報を dict にして返す
        output = []
        for box in boxes:
            b = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label_name = COCO_LABELS[cls_id] if cls_id < len(COCO_LABELS) else "unknown"
            
            output.append({
                "xmin": b[0],
                "ymin": b[1],
                "xmax": b[2],
                "ymax": b[3],
                "confidence": conf,
                "label": label_name
            })
        return output
