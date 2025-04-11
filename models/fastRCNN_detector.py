import torch
import torchvision
from torchvision import transforms
from PIL import Image

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


class FasterRCNNDetector:
    def __init__(self, threshold=0.5):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.threshold = threshold
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.labels = torchvision.datasets.CocoDetection.classes if hasattr(torchvision.datasets.CocoDetection, 'classes') else [str(i) for i in range(91)]

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(input_tensor)[0]
        
        results = []
        for box, score, label in zip(outputs['boxes'], outputs['scores'], outputs['labels']):
            if score >= self.threshold:
                label_name = COCO_LABELS[label.item()] if label.item() < len(COCO_LABELS) else "unknown"
                results.append({
                    "xmin": box[0].item(),
                    "ymin": box[1].item(),
                    "xmax": box[2].item(),
                    "ymax": box[3].item(),
                    "confidence": score.item(),
                    "label": label_name
                })
        return results
