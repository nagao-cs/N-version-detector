from models.fastRCNN_detector import FasterRCNNDetector
from models.yolov5_detector import YOLOv5Detector
from models.yolov8_detector import YOLOv8Detector
from models.SSD_detector import SSDDetector
from utils.visualize import draw_boxes, show_side_by_side, show_matched_boxes   
from utils.iou import match_detections

IMAGE_PATH = 'sample.png'

# モデル初期化
yolov8 = YOLOv8Detector()
yolov5 = YOLOv5Detector()
fasterrcnn = FasterRCNNDetector()
ssd = SSDDetector()

# 推論
yolov8_result = yolov8.predict(IMAGE_PATH)
yolov5_result = yolov5.predict(IMAGE_PATH)
fasterrcnn_result = fasterrcnn.predict(IMAGE_PATH)
ssd_result = ssd.predict(IMAGE_PATH)

# 描画
img_yolov8 = draw_boxes(IMAGE_PATH, yolov8_result, color=(0, 255, 0), label='YOLOv8')
img_yolov5 = draw_boxes(IMAGE_PATH, yolov5_result, color=(0, 0, 255), label='YOLOv5')
img_fasterrcnn = draw_boxes(IMAGE_PATH, fasterrcnn_result, color=(0, 0, 255), label='FRCNN')
img_ssd = draw_boxes(IMAGE_PATH, ssd_result, color=(0, 255, 0), label='SSD')

# 表示
show_side_by_side(img_yolov8, img_yolov5, img_ssd, title1='YOLOv8', title2='YOLOv5', title3='SSD')
matches = match_detections(yolov8_result, yolov5_result, iou_threshold=0.5)
print(f"matches: {len(matches)}件")
for box1, box2, iou in matches:
    print(f"box1: {box1['label']}, box2: {box2['label']}, iou: {iou:.2f}")

show_matched_boxes(IMAGE_PATH, matches, color=(0, 255, 0), label='matched')
