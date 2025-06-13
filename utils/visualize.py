import cv2
import matplotlib.pyplot as plt

def draw_match_bboxes(image, bboxes):
    for bbox in bboxes:
        if bbox['num_match'] == 'all':
            color = (0, 0, 255) #赤
            label = 'unanimous'
        elif bbox['num_match'] == 'majority':
            color = (0, 255, 0) #緑
            label = 'majority'
        else:
            color = (255, 0, 0)
            label = 'single'
        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label, (xmin, ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return image

def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    return image