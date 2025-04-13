from flask import Flask, render_template, request, send_from_directory
from models.yolov8_detector import YOLOv8Detector
from models.yolov5_detector import YOLOv5Detector
from models.fastRCNN_detector import FasterRCNNDetector
from models.SSD_detector import SSDDetector
from utils.visualize import draw_boxes, show_side_by_side
from utils.iou import match_detections
import os

app = Flask(__name__)

# モデル初期化
yolov8 = YOLOv8Detector()
yolov5 = YOLOv5Detector()
ssd = SSDDetector()

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # アップロード画像の取得
        file = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.png')
        file.save(path)

        # 検出
        yolov8_result = yolov8.predict(path)
        yolov5_result = yolov5.predict(path)
        ssd_result = ssd.predict(path)
        matches = match_detections(yolov8_result, yolov5_result)
        matched_boxes = [m[0] for m in matches]

        # 可視化 & 保存
        output_img = draw_boxes(path, matched_boxes, color=(0, 255, 0), label='一致')
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        import cv2
        cv2.imwrite(output_path, output_img)

        return render_template('index.html', result_img='output.png')

    return render_template('index.html', result_img=None)

@app.route('/static/<filename>')
def static_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
