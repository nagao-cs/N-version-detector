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

def match_detections(*results, iou_threshold=0.7):
    num_version = len(results)
    bboxes = list()
    
    if num_version == 1:
        for bbox in results[0]:
            bbox = {
                'xmin': bbox['xmin'],
                'ymin': bbox['ymin'],
                'xmax': bbox['xmax'],
                'ymax': bbox['ymax'],
                'num_match': 'all',
                'label': bbox['label']
            }
            bboxes.append(bbox)
    elif num_version == 2:
        result1, result2 = results[0], results[1]
        for bbox1 in result1:
            flag = False
            for bbox2 in result2:
                if (not flag) and (bbox1['label'] == bbox2['label']) and (compute_iou(bbox1, bbox2) > iou_threshold):
                    bbox = {
                        'xmin': (bbox1['xmin']+bbox2['xmin'])/2,
                        'ymin': (bbox1['ymin']+bbox2['ymin'])/2,
                        'xmax': (bbox1['xmax']+bbox2['xmax'])/2,
                        'ymax': (bbox1['ymax']+bbox2['ymax'])/2,
                        'num_match': 'all',
                        'label': bbox1['label']
                    }
                    flag = True
                    bboxes.append(bbox)
                else:
                    bbox = {
                        'xmin': bbox2['xmin'],
                        'ymin': bbox2['ymin'],
                        'xmax': bbox2['xmax'],
                        'ymax': bbox2['ymax'],
                        'num_match': '1',
                        'label': bbox2['label']
                    }
                    bboxes.append(bbox)
            if not flag:
                bbox = {
                        'xmin': bbox1['xmin'],
                        'ymin': bbox1['ymin'],
                        'xmax': bbox1['xmax'],
                        'ymax': bbox1['ymax'],
                        'num_match': '1',
                        'label': bbox1['label']
                    }
                bboxes.append(bbox)
    elif num_version == 3:
        result1, result2, result3 = results
        used2 = set()
        used3 = set()
        for bbox1 in result1:
            matched = False
            # まず1と2を調べる
            for i2, bbox2 in enumerate(result2):
                if bbox1['label'] != bbox2['label'] or i2 in used2:
                    continue
                iou12 = compute_iou(bbox1, bbox2)
                if iou12 > iou_threshold:
                    # 1と2が一致した場合は3も調べる
                    for i3, bbox3 in enumerate(result3):
                        if bbox1['label'] != bbox3['label'] or i3 in used3:
                            continue
                        iou13 = compute_iou(bbox1, bbox3)
                        iou23 = compute_iou(bbox2, bbox3)
                        if iou13 > iou_threshold and iou23 > iou_threshold:
                            # 3つとも一致
                            bbox = {
                                'xmin': (bbox1['xmin'] + bbox2['xmin'] + bbox3['xmin']) / 3,
                                'ymin': (bbox1['ymin'] + bbox2['ymin'] + bbox3['ymin']) / 3,
                                'xmax': (bbox1['xmax'] + bbox2['xmax'] + bbox3['xmax']) / 3,
                                'ymax': (bbox1['ymax'] + bbox2['ymax'] + bbox3['ymax']) / 3,
                                'num_match': 'all',
                                'label': bbox1['label']
                            }
                            bboxes.append(bbox)
                            used2.add(i2)
                            used3.add(i3)
                            matched = True
                            break
                if matched:
                    break
                # 1と2のみ一致
                bbox = {
                    'xmin': (bbox1['xmin'] + bbox2['xmin']) / 2,
                    'ymin': (bbox1['ymin'] + bbox2['ymin']) / 2,
                    'xmax': (bbox1['xmax'] + bbox2['xmax']) / 2,
                    'ymax': (bbox1['ymax'] + bbox2['ymax']) / 2,
                    'num_match': 'majority',
                    'label': bbox1['label']
                }
                bboxes.append(bbox)
                used2.add(i2)
                matched = True
                break
            if not matched:
                # 1と2が一致しなかった場合、1と3を調べる
                for i3, bbox3 in enumerate(result3):
                    if bbox1['label'] != bbox3['label'] or i3 in used3:
                        continue
                    iou13 = compute_iou(bbox1, bbox3)
                    if iou13 > iou_threshold:
                        bbox = {
                            'xmin': (bbox1['xmin'] + bbox3['xmin']) / 2,
                            'ymin': (bbox1['ymin'] + bbox3['ymin']) / 2,
                            'xmax': (bbox1['xmax'] + bbox3['xmax']) / 2,
                            'ymax': (bbox1['ymax'] + bbox3['ymax']) / 2,
                            'num_match': 'majority',
                            'label': bbox1['label']
                        }
                        bboxes.append(bbox)
                        used3.add(i3)
                        matched = True
                        break
            if not matched:
                # どれとも一致しない場合
                bbox = {
                    'xmin': bbox1['xmin'],
                    'ymin': bbox1['ymin'],
                    'xmax': bbox1['xmax'],
                    'ymax': bbox1['ymax'],
                    'num_match': '1',
                    'label': bbox1['label']
                }
                bboxes.append(bbox)
        # result2, result3の未使用bboxも追加
        for i2, bbox2 in enumerate(result2):
            if i2 not in used2:
                bbox = {
                    'xmin': bbox2['xmin'],
                    'ymin': bbox2['ymin'],
                    'xmax': bbox2['xmax'],
                    'ymax': bbox2['ymax'],
                    'num_match': '1',
                    'label': bbox2['label']
                }
                bboxes.append(bbox)
        for i3, bbox3 in enumerate(result3):
            if i3 not in used3:
                bbox = {
                    'xmin': bbox3['xmin'],
                    'ymin': bbox3['ymin'],
                    'xmax': bbox3['xmax'],
                    'ymax': bbox3['ymax'],
                    'num_match': '1',
                    'label': bbox3['label']
                }
                bboxes.append(bbox)
    
    return bboxes