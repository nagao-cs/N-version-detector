from ultralytics import YOLO
from .coco_labels import COCO_LABELS

class YOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = YOLO(model_path)

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


