from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class DetectionResult(db.Model):
    __tablename__ = 'result'
    id = db.Column(db.String(200), primary_key=True)
    original_image = db.Column(db.String(200))         # 元画像ファイル名
    result_img_normal = db.Column(db.String(200))      # 元画像の検出結果ファイル名
    result_img_gray = db.Column(db.String(200))        # グレースケール検出結果ファイル名
    result_img_reverse = db.Column(db.String(200))     # 反転画像検出結果ファイル名
    matched_result = db.Column(db.String(200))         #統合後の画像のファイル名