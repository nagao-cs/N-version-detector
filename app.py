from flask import Flask, render_template, request, send_from_directory, redirect
from models.yolov8_detector import YOLOv8Detector
from models.yolov5_detector import YOLOv5Detector
# from models.fastRCNN_detector import FasterRCNNDetector
# from models.SSD_detector import SSDDetector
from utils.visualize import draw_match_bboxes, draw_bboxes
from utils.iou import match_detections
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import cv2



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

UPLOAD_FOLDER = 'input'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PROCESSED_FOLDER = 'processed'
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['PROCESSED'] = PROCESSED_FOLDER
OUPUT_FOLDER = 'output'
os.makedirs(OUPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUPUT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_version = int(request.form["num_version"])
        dt_now = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # アップロード画像の取得
        file = request.files['image']
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{dt_now}.png')
        file.save(input_path)
        
        if num_version >= 1:
            normal_image = cv2.imread(input_path)
            normal_path = os.path.join(app.config['PROCESSED'], f'{dt_now}_normal.png')
            cv2.imwrite(normal_path, normal_image)
        # グレースケール画像
        if num_version >= 2:
            gray_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            gray_path = os.path.join(app.config['PROCESSED'], f'{dt_now}_gray.png')
            cv2.imwrite(gray_path, gray_image)
        # 色を反転した画像
        if num_version >= 3:
            reverse_image = cv2.bitwise_not(normal_image)
            reverse_path = os.path.join(app.config['PROCESSED'], f'{dt_now}_reverse.png')
            cv2.imwrite(reverse_path, reverse_image)

        # 検出
        normal_result = yolov8.predict(normal_image)
        normal_bbox_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{dt_now}_normal.png')
        cv2.imwrite(normal_bbox_path, draw_bboxes(normal_image, normal_result))
        if num_version >= 2:
            gray_result = yolov8.predict(gray_image)
            gray_bbox_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{dt_now}_gray.png')
            cv2.imwrite(gray_bbox_path, draw_bboxes(gray_image, gray_result))
        if num_version >= 3:
            reverse_result = yolov8.predict(reverse_image)
            reverse_bbox_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{dt_now}_reverse.png')
            cv2.imwrite(reverse_bbox_path, draw_bboxes(reverse_image, reverse_result))

        # マッチング
        if num_version == 1:
            matched_bboxes = match_detections(normal_result)
        elif num_version == 2:
            matched_bboxes = match_detections(normal_result, gray_result)
        elif num_version >= 3:
            matched_bboxes = match_detections(normal_result, gray_result, reverse_result)
        
        matched_boxes = [bbox for bbox in matched_bboxes]

        # 可視化 & 保存
        image = cv2.imread(input_path)
        output_img = draw_match_bboxes(image, matched_boxes)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{dt_now}.png')
        cv2.imwrite(output_path, output_img)
        
        # DBに保存
        new_record = DetectionResult(image_filename=f'{dt_now}.png')
        db.session.add(new_record)
        db.session.commit()
        
        return render_template(
            'index.html',
            result_img_normal=f"{dt_now}_normal.png",
            result_img_gray=f"{dt_now}_gray.png" if num_version >= 2 else None,
            result_img_reverse=f"{dt_now}_reverse.png" if num_version >= 3 else None,
            result_img=f'{dt_now}.png'
        )

    return render_template('index.html', result_img=None)

@app.route('/output/<filename>')
def output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

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
