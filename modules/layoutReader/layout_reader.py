import json

import torch
from collections import defaultdict

from modules.layoutReader.model import LayoutLMv3ForBboxClassification
from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox

CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2


def BboxesMasks(boxes):
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] + [1] * len(boxes) + [1]
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }


def scale_bbox(bbox, width, height):
    x_scale = min(1000.0 / width, 1)
    y_scale = min(1000.0 / height, 1)

    boxes = []
    print(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(bbox)}")
    for left, top, right, bottom in bbox:
        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
                1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f"Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}"
        boxes.append([left, top, right, bottom])
    return boxes


def decode(logits, length):
    logits = logits[1: length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        for order, idxes in order_to_idxes.items():
            idxes_to_logit = {}
            for idx in idxes:
                idxes_to_logit[idx] = logits[idx, order]
            idxes_to_logit = sorted(
                idxes_to_logit.items(), key=lambda x: x[1], reverse=True
            )
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()
    return ret


def layout_reader_sort(layout_boxes: list[LayoutBox], model_path: str, width: int, height: int) -> (
        list[LayoutBox], list[int]):
    """
  对布局盒子进行排序或其他处理。

  :param layout_boxes: LayoutBox对象的列表
  :param model_path: 模型路径的字符串
  :param width: 宽度的整数值
  :param height: 高度的整数值
  """
    bboxes = [layout_box.bbox for layout_box in layout_boxes]
    scaled_boxes = scale_bbox(bboxes, width, height)
    model = LayoutLMv3ForBboxClassification.from_pretrained(model_path)
    inputs = BboxesMasks(scaled_boxes)
    logits = model(**inputs).logits.cpu().squeeze(0)
    orders = decode(logits, len(scaled_boxes))
    layout_boxes = [layout_boxes[i] for i in orders]
    return layout_boxes, orders


if __name__ == '__main__':
    bboxes = [[584, 0, 595, 1], [35, 120, 89, 133],
              [35, 140, 75, 152]]
    model_path = f"../../models/LayoutReader"
    # print(layout_reader_sort(bboxes, model_path, 400, 400))
