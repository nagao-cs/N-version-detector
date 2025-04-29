def compute_iou(box1, box2):
    # box: dict with xmin, ymin, xmax, ymax
    xA = max(box1['xmin'], box2['xmin'])
    yA = max(box1['ymin'], box2['ymin'])
    xB = min(box1['xmax'], box2['xmax'])
    yB = min(box1['ymax'], box2['ymax'])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    box1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    box2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def match_detections(list1, list2, list3, iou_threshold=0.5):
    matches = []
    for box1 in list1:
        for box2 in list2:
            if box1['label'] == box2['label']:
                for box3 in list3:
                    if box1['label'] == box3['label']:
                        iou_1 = compute_iou(box1, box2)
                        iou_2 = compute_iou(box1, box3)
                        iou_3 = compute_iou(box2, box3)
                        if iou_1 > iou_threshold and iou_2 > iou_threshold and iou_3 > iou_threshold:
                            iou = (iou_1 + iou_2 + iou_3) / 3
                            box = {
                                "xmin": (box1['xmin'] + box2['xmin'] + box3['xmin']) / 3,
                                "ymin": (box1['ymin'] + box2['ymin'] + box3['ymin']) / 3,
                                "xmax": (box1['xmax'] + box2['xmax'] + box3['xmax']) / 3,
                                "ymax": (box1['ymax'] + box2['ymax'] + box3['ymax']) / 3,
                                "confidence": (box1['confidence'] + box2['confidence'] + box3['confidence']) / 3,
                                "label": box1['label']
                            }
                            matches.append((box, iou))
    return matches