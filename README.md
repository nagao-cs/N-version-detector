# N_version_Detector

<<<<<<< HEAD
## 次にやること
- CRUD操作の実装
- 一致した検出結果だけでなく一致率や各モデルの検出結果など、様々な情報を出せるようにする
- デプロイについて勉強
- フロントエンドをもっときれいに
- ある程度できたらReactについて勉強
- ある程度できたら機械学習の実装や学習など細かなとこまで勉強

=======
>>>>>>> 761fef28b81cbce1951dac033ec4321650b45cb8
## 概要
複数の物体検出モデル（YOLOv5, YOLOv8, SSD, Faster R-CNN）を使用して、画像内の物体を検出し、各モデルの検出結果を比較・可視化するWebアプリケーションです。

## 主な機能
- 画像のアップロードと物体検出
- 複数の物体検出モデルの同時実行
- 検出結果の可視化と比較

## インストール方法
1. リポジトリをクローン
```bash
git clone https://github.com/nagao-cs/N-version-detector/
cd N_version_detector
```

2. 仮想環境の作成と有効化
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

## 使用方法
1. アプリケーションの起動
```bash
python app.py
```

2. ローカルホストにアクセス
3. 画像をアップロードして物体検出を実行

## 必要な環境
- Python 3.8以上
- PyTorch
- Flask
- OpenCV
- その他 requirements.txt に記載のパッケージ

## 依存関係
- YOLOv5
- YOLOv8
- SSD
- Faster R-CNN

## ライセンス
MIT License

## 次にやること
- 一致した検出結果だけでなく一致率や各モデルの検出結果など、様々な情報を出せるようにする
- デプロイについて勉強
- フロントエンドをもっときれいに
- ある程度できたらReactについて勉強
- ある程度できたら機械学習の実装や学習など細かなとこまで勉強
