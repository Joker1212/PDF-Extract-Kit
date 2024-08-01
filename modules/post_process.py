import re

from paddleocr import PPStructure

from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox, id2names


def layout_rm_equation(layout_res):
    rm_idxs = []
    for idx, ele in enumerate(layout_res['layout_dets']):
        if ele['category_id'] == 10:
            rm_idxs.append(idx)

    for idx in rm_idxs[::-1]:
        del layout_res['layout_dets'][idx]
    return layout_res


def get_croped_image(image_pil, bbox):
    x_min, y_min, x_max, y_max = bbox
    croped_img = image_pil.crop((x_min, y_min, x_max, y_max))
    return croped_img


def latex_rm_whitespace(s: str):
    """Remove unnecessary whitespace from LaTeX code.
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s


def is_contained(box1, box2, threshold=0.1):
    """
    计算box1是否包含了box2
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return False
    # 计算box2的总面积
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 计算box1和box2的交集
    intersect_x1 = max(b1_x1, b2_x1)
    intersect_y1 = max(b1_y1, b2_y1)
    intersect_x2 = min(b1_x2, b2_x2)
    intersect_y2 = min(b1_y2, b2_y2)

    # 计算交集的面积
    intersect_area = max(0, intersect_x2 - intersect_x1) * max(0, intersect_y2 - intersect_y1)

    # 计算box2在box1外面的面积
    outside_area = b2_area - intersect_area

    # 计算外面的面积占box2总面积的比例
    ratio = outside_area / b2_area if b2_area > 0 else 0

    # 判断比例是否大于阈值
    return ratio < threshold


def calculate_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    # 不相交直接退出检测
    if b1_x2 < b2_x1 or b1_x1 > b2_x2 or b1_y2 < b2_y1 or b1_y1 > b2_y2:
        return 0.0
    # 计算交集
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算并集
    area_box1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_box2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area_box1 + area_box2 - inter_area

    # 避免除零错误，如果区域小到乘积为0,认为是错误识别，直接去掉
    if union_area == 0:
        return 1
        # 检查完全包含
    iou = inter_area / union_area
    return iou


# 将包含关系和重叠关系的box进行过滤，只保留一个
def filter_consecutive_boxes(layout_boxes: list[LayoutBox], iou_threshold=0.92) -> (list[LayoutBox], list[int]):
    """
    检测布局框列表中包含关系和重叠关系，只保留一个
    LayoutBox.bbox: (xmin,ymin,xmax,ymax)
    """
    boxes = [layout_box.bbox for layout_box in layout_boxes]
    if len(boxes) <= 1:
        return boxes
    for box1 in boxes:
        if not box1:
            continue
        for box2 in boxes:
            if not box2:
                continue
            if box1 == box2:
                continue
            if is_contained(box1, box2) or calculate_iou(box1, box2) > iou_threshold:
                i = boxes.index(box2)
                boxes[i] = None
    filtered_boxes = [layout_boxes[i] for i, box in enumerate(boxes) if box is not None]
    idx = [i for i, box in enumerate(boxes) if box is not None]
    return filtered_boxes, idx


def pe_res_trans_2_layout_box(pe_layout_res) -> list[LayoutBox]:
    label_bboxes = []
    for idx, res in enumerate(pe_layout_res['layout_dets']):
        xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
        xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
        bbox = LayoutBox((xmin, ymin, xmax, ymax), id2names[res['category_id']], res['score'])
        label_bboxes.append(bbox)
    return label_bboxes


def layout_box_trans_2_pe_res(layout_boxes: list[LayoutBox]) -> dict:
    pe_layout_res = {}
    pe_layout_dets = []
    for idx, bbox in enumerate(layout_boxes):
        pe_layout_dets.append({
            'poly':
                [bbox.bbox[0], bbox.bbox[1],
                 bbox.bbox[2], bbox.bbox[1],
                 bbox.bbox[2], bbox.bbox[3],
                 bbox.bbox[0], bbox.bbox[3]],
            'category_id': id2names.index(bbox.label),
            'score': bbox.score
        })
    pe_layout_res['layout_dets'] = pe_layout_dets
    return pe_layout_res


def trans_2_layout_box(paddle_layout_res) -> list[LayoutBox]:
    label_bboxes = []
    for idx, paddle_dict in enumerate(paddle_layout_res):
        bbox = LayoutBox(paddle_dict['bbox'], paddle_dict['type'])
        label_bboxes.append(bbox)
    return label_bboxes


def header_footer_fix_by_paddleocr(img_array, label_boxes: list[LayoutBox], iou_threshold=0.5, ppstructure=None):
    # 修正识别为title的页眉页脚信息
    if not ppstructure:
        ppstructure = PPStructure(table=False, ocr=False, show_log=True)
    paddle_layout_res = ppstructure(img_array)
    paddle_label_bboxes = trans_2_layout_box(paddle_layout_res)
    # 找到paddle识别到的页眉页脚
    paddle_label_bboxes = [paddle_label_bboxes[i] for i in range(len(paddle_label_bboxes)) if
                           paddle_label_bboxes[i].label == 'header' or paddle_label_bboxes[i].label == 'footer']
    for i, box in enumerate(label_boxes):
        for paddle_box in paddle_label_bboxes:
            if calculate_iou(paddle_box.bbox, box.bbox) >= iou_threshold or is_contained(paddle_box.bbox, box.bbox):
                label_boxes[i].label = 'abandon'
    return label_boxes


def layout_abandon_fix_to_text(layout_boxes: list[LayoutBox]) -> (list[LayoutBox], list[int]):
    """
    修改布局框列表中元素的label属性，以修正页眉页脚的错误识别。

    当遇到label为'abandon'的元素，但其后一个元素的label不是'abandon'时，
    将当前元素的label属性修改为'plain_text'。

    参数:
    layout_boxes (List[LayoutBox]): 包含LayoutBox对象的列表，每个对象代表页面上的一个布局框。

    返回:
    Tuple[List[LayoutBox], List[int]]:
        第一个元素是修改后的LayoutBox对象列表；
        第二个元素是包含所有被修改元素索引的列表。
    """
    modified_indices = []

    # 遍历列表中的元素，除了最后一个元素
    for i in range(len(layout_boxes) - 1):
        # 检查当前元素的label是否为'abandon'
        if layout_boxes[i].label == 'abandon':
            # 检查下一个元素的label是否不是'abandon'
            if layout_boxes[i + 1].label != 'abandon':
                # 将当前元素的label设置为'plain_text'
                layout_boxes[i].label = 'plain_text'
                modified_indices.append(i)  # 记录修改的索引

    return layout_boxes, modified_indices

# def layout_title_fix_to_abandan()
