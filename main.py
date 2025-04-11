from models.yolo_detector import YOLODetector
from models.fastRCNN_detector import FasterRCNNDetector
from utils.visualize import draw_boxes, show_side_by_side

IMAGE_PATH = 'sample.png'

# モデル初期化
yolo = YOLODetector()
fasterrcnn = FasterRCNNDetector()

# 推論
yolo_result = yolo.predict(IMAGE_PATH)
fasterrcnn_result = fasterrcnn.predict(IMAGE_PATH)

# 描画
img_yolo = draw_boxes(IMAGE_PATH, yolo_result, color=(0, 255, 0), label='YOLO')
img_fasterrcnn = draw_boxes(IMAGE_PATH, fasterrcnn_result, color=(0, 0, 255), label='FRCNN')

# 表示
show_side_by_side(img_yolo, img_fasterrcnn)

