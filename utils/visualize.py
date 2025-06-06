import cv2
import matplotlib.pyplot as plt

def draw_boxes(image_path, detections, color=(255, 0, 0), label=''):
    image = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label}:{det.get('label', '')}"
        #bboxの10px上にラベル(物体のクラス)を表示
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def show_side_by_side(img1, img2, img3, title1='Model A', title2='Model B', title3='Model C'):
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title1)
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title2)
    axs[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    axs[2].set_title(title3)
    plt.show()
    plt.savefig('result.png')

#一致したボックスを描画
def show_matched_boxes(image_path, matches, color=(0, 255, 0), label=''):
    image = cv2.imread(image_path)
    for match in matches:
        box1, box2, iou = match
        x1 = int((int(box1['xmin']) + int(box2['xmin'])) / 2)
        y1 = int((int(box1['ymin']) + int(box2['ymin'])) / 2)
        x2 = int((int(box1['xmax']) + int(box2['xmax'])) / 2)
        y2 = int((int(box1['ymax']) + int(box2['ymax'])) / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label}:{iou:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('result.png', image)


