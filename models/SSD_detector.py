import torchvision
from torchvision.models.detection import ssd300_vgg16
from torchvision import transforms
import torch
from PIL import Image
import COCO_LABELS

class SSDDetector:
    def __init__(self, threshold=0.5):
        self.model = ssd300_vgg16(pretrained=True)
        self.model.eval()
        self.threshold = threshold
        
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
    
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            ])
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
        outputs = []
        for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
            if score >= self.threshold:
                label_name = COCO_LABELS[label.item()] if label.item() < len(COCO_LABELS) else "unknown"
                outputs.append({
                    'xmin': box[0],
                    'ymin': box[1],
                    'xmax': box[2],
                    'ymax': box[3],
                    'confidence': score,
                    'label': label_name,
                })
        return outputs
        
        