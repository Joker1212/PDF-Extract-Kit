
id2names = ["title", "text", "header", "figure", "figure_caption", "table", "table_caption","table_footnote", "equation",  "reference"]
continuealbe_labels=['text', 'table']
class LayoutBox:
    def __init__(self, bbox, label, score=1.0, box_type='single', merged_bbox=None):
        """
        初始化LayoutBox类

        :param bbox: 四元组 (xmin, ymin, xmax, ymax)，表示边界框的位置
        :param label: str 类型，表示布局元素的标签
        :param score: float 类型，标识置信度
        :param box_type: str 类型，标识是普通单个布局元素还是多个布局元素合成的, single, merge
        :param merged_bbox: 列表 类型，每个元素是四元组 (xmin, ymin, xmax, ymax)，表示边界框的位置
        """
        self.bbox = bbox
        self.label = label
        self.score = score
        self.box_type = box_type
        self.merged_bbox = merged_bbox
    def __repr__(self):
        """
        返回LayoutBox对象的字符串表示
        """
        return f"LayoutBox(bbox=({self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]}), label='{self.label}')"

    def to_dict(self):
        return {'bbox':self.bbox, 'label':self.label, 'score':self.score, 'box_type':self.box_type, 'merged_bbox':self.merged_bbox}


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
            # if is_contained(box1, box2) or calculate_iou(box1, box2) > iou_threshold:
            if calculate_iou(box1, box2) > iou_threshold:
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


def sorted_layout_boxes(res: list[LayoutBox], w):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        res(list):ppstructure results
    return:
        sorted results(list)
    """
    # res = [layout_boxes[i].bbox.bbox for i in range(len(layout_boxes))]
    num_boxes = len(res)
    if num_boxes == 1:
        # res[0]["layout"] = "single"
        return res

    sorted_boxes = sorted(res, key=lambda x: (x.bbox[1], x.bbox[0]))
    _boxes = list(sorted_boxes)

    new_res = []
    res_left = []
    res_right = []
    i = 0

    while i < num_boxes:
        box_len = max(0, _boxes[i].bbox[2] - _boxes[i].bbox[0])
        if box_len == 0:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
            continue
        if i >= num_boxes:
            break
        if i == num_boxes - 1:
            if (
                    _boxes[i].bbox[1] > _boxes[i - 1].bbox[3]
                    and _boxes[i].bbox[0] < w / 2
                    and _boxes[i].bbox[2] > w / 2
            ):
                new_res += res_left
                new_res += res_right
                new_res.append(_boxes[i])
            else:
                if _boxes[i].bbox[2] > w / 2:
                    res_right.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
                elif _boxes[i].bbox[0] < w / 2:
                    res_left.append(_boxes[i])
                    new_res += res_left
                    new_res += res_right
            # res_left = []
            # res_right = []
            break
        #   box两边距离中线偏移不大，则认为是居中的布局
        elif _boxes[i].bbox[0] < w / 2 and _boxes[i].bbox[2] > w / 2 and (
                _boxes[i].bbox[2] - w / 2) / box_len < 0.65 and (w / 2 - _boxes[i].bbox[0]) / box_len < 0.65:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
        elif _boxes[i].bbox[0] < w / 4 and _boxes[i].bbox[2] < 3 * w / 4:
            res_left.append(_boxes[i])
            i += 1
        elif _boxes[i].bbox[0] > w / 4 and _boxes[i].bbox[2] > w / 2:
            res_right.append(_boxes[i])
            i += 1
        else:
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1
    return new_res


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
        if layout_boxes[i].label == 'abandon' and i != 0:
            # 检查下一个元素的label是否不是'abandon'
            if layout_boxes[i + 1].label != 'abandon':
                # 将当前元素的label设置为'plain_text'
                layout_boxes[i].label = 'plain_text'
                modified_indices.append(i)  # 记录修改的索引

    return layout_boxes, modified_indices


def handle_page_between_box_merge(all_label_boxes):
    """
    将跨页的分块整合起来，可能出现某页为空
    """
    for i in range(len(all_label_boxes) - 1):
        if not all_label_boxes[i]:
            continue
        cur = all_label_boxes[i][-1]
        next = all_label_boxes[i + 1][0]
        # 类型为表格或普通文本，且当前检测框与下一个检测框之间的距离大于当前检测框长度
        if cur.label == next.label and cur.label in continuealbe_labels:
            if cur.box_type == 'merge' and next.box_type == 'merge':
                cur.merged_bbox.extend(next.merged_bbox)
                merge_label_box = cur
            elif cur.box_type == 'merge':
                cur.merged_bbox.append(next.bbox)
                merge_label_box = cur
            elif next.box_type == 'merge':
                next.merged_bbox.insert(0, cur.bbox)
                merge_label_box = next
            else:
                merge_label_box = LayoutBox(label=cur.label, score=cur.score, bbox=None, box_type='merge',
                                            merged_bbox=[cur.bbox, next.bbox])

            all_label_boxes[i][-1] = merge_label_box
            all_label_boxes[i + 1].pop(0)
    return all_label_boxes


def handle_page_inner_box_merge(label_boxes):
    """
    将多列pdf可能导致的文本或表格分段整合成一个merge_box
    """
    if len(label_boxes) <= 1:
        return label_boxes
    for i in range(len(label_boxes) - 1):
        cur = label_boxes[i]
        if not cur:
            continue
        next = label_boxes[i + 1]
        cur_box_len = cur.bbox[2] - cur.bbox[0]
        # 类型为表格或普通文本，且当前检测框与下一个检测框之间的距离大于当前检测框长度
        if cur.label == next.label and cur.label in continuealbe_labels and next.bbox[0] - cur.bbox[0] > cur_box_len:
            merge_label_box = LayoutBox(label=cur.label, score=cur.score, bbox=None, box_type='merge',
                                        merged_bbox=[cur.bbox, next.bbox])
            label_boxes[i] = merge_label_box
            label_boxes[i + 1] = None
            i += 1
    return [box for box in label_boxes if box is not None]