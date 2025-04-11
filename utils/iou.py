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

def match_detections(list1, list2, iou_threshold=0.5):
    matches = []
    for det1 in list1:
        for det2 in list2:
            print(f"det1: {det1['label']}, det2: {det2['label']}")
            if det1['label'] == det2['label']:
                iou = compute_iou(det1, det2)
                print(f"iou: {iou:.2f}, {det1['label']}, {det2['label']}")
                if iou >= iou_threshold:
                    matches.append((det1, det2, iou))
    return matches
