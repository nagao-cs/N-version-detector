import sys
import os
from pathlib import Path

# yolov5 ディレクトリを Python モジュール検索パスに追加
YOLO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5'))
if YOLO_PATH not in sys.path:
    sys.path.append(YOLO_PATH)

import torch

class YOLODetector:
    def __init__(self, weights='yolov5s.pt'):
        # yolov5 フォルダを直接使うことで models.common の import エラーを回避
        self.model = torch.hub.load(YOLO_PATH, 'custom', path=weights, source='local')

    def predict(self, image_path):
        results = self.model(image_path)
        return results.pandas().xyxy[0].to_dict('records')
