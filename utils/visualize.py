import cv2
import matplotlib.pyplot as plt

def draw_boxes(image_path, detections, color=(255, 0, 0), label=''):
    image = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label}:{det.get('label', '')}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def show_side_by_side(img1, img2, title1='Model A', title2='Model B'):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title1)
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title2)
    plt.show()
