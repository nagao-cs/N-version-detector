import torch
import torchvision
from torchvision import transforms
from PIL import Image

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
                results.append({
                    "xmin": box[0].item(),
                    "ymin": box[1].item(),
                    "xmax": box[2].item(),
                    "ymax": box[3].item(),
                    "confidence": score.item(),
                    "label": str(label.item())  # ラベル名変換は後で対応
                })
        return results
