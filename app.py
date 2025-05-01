from flask import Flask, render_template, request, send_from_directory, redirect

from models.yolov8_detector import YOLOv8Detector
from models.yolov5_detector import YOLOv5Detector
# from models.fastRCNN_detector import FasterRCNNDetector
# from models.SSD_detector import SSDDetector
from utils.visualize import draw_boxes, show_side_by_side
from utils.iou import match_detections
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# データモデル
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
# モデル初期化
yolov8 = YOLOv8Detector()
yolov5 = YOLOv5Detector()
yolov5_2 = YOLOv5Detector()

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
        yolov5_2_result = yolov5_2.predict(path)
        print("YOLOv8 Result:", yolov8_result)
        matches = match_detections(yolov8_result, yolov5_result, yolov5_2_result)
        matched_boxes = [m[0] for m in matches]

        # 可視化 & 保存
        output_img = draw_boxes(path, matched_boxes, color=(0, 255, 0), label='match')
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
        import cv2
        cv2.imwrite(output_path, output_img)
        
        # DBに保存
        new_record = DetectionResult(image_filename='input.jpg')  # ファイル名は動的でもOK
        db.session.add(new_record)
        db.session.commit()
        
        return render_template('index.html', result_img='output.png')

    return render_template('index.html', result_img=None)

@app.route('/static/<filename>')
def static_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
def history():
    results = DetectionResult.query.order_by(DetectionResult.created_at.desc()).all()
    return render_template('history.html', results=results)

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    record = DetectionResult.query.get_or_404(id)

    # 画像ファイルも削除
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], record.image_filename)
    if os.path.exists(image_path):
        os.remove(image_path)

    # DBから削除
    db.session.delete(record)
    db.session.commit()

    return redirect('/history')


if __name__ == '__main__':
    app.run(debug=True)
