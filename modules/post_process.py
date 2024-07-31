import re


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


def is_contained(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    return b2_x1 >= b1_x1 and b2_y1 >= b1_y1 and b2_x2 <= b1_x2 and b2_y2 <= b1_y2


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
def filter_consecutive_boxes(boxes, iou_threshold=0.92):
    if len(boxes) <= 1:
        return boxes
    filtered_boxes = []
    # 从后向前遍历列表
    i = len(boxes) - 2  # 开始于倒数第二个元素
    current_box = boxes[i + 1]
    previous_box = boxes[i]
    cur_idx = i + 1
    idx = []
    while i >= 0:
        # 检查当前框是否包含或重叠于前一个框,或者包含了前一个框
        if not is_contained(current_box, previous_box) and calculate_iou(current_box, previous_box) <= iou_threshold:
            filtered_boxes.insert(0, current_box)
            current_box = previous_box
            idx.insert(0, cur_idx)
            cur_idx = i
        i -= 1
        previous_box = boxes[i]
    filtered_boxes.insert(0, current_box)
    idx.insert(0, cur_idx)
    return filtered_boxes, idx
