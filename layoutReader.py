import os

import cv2
import torch
from transformers import LayoutLMv3ForTokenClassification
from collections import defaultdict
from typing import List, Dict
CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2
def boxes2inputs(boxes: List[List[int]]) -> Dict[str, torch.Tensor]:
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] + [1] * len(boxes) + [1]
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }


def prepare_inputs(
    inputs: Dict[str, torch.Tensor], model: LayoutLMv3ForTokenClassification
) -> Dict[str, torch.Tensor]:
    ret = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        ret[k] = v
    return ret


def parse_logits(logits: torch.Tensor, length: int) -> List[int]:
    """
    parse logits to orders

    :param logits: logits from model
    :param length: input length
    :return: orders
    """
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        # filter idxes len > 1
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        # filter
        for order, idxes in order_to_idxes.items():
            # find original logits of idxes
            idxes_to_logit = {}
            for idx in idxes:
                idxes_to_logit[idx] = logits[idx, order]
            idxes_to_logit = sorted(
                idxes_to_logit.items(), key=lambda x: x[1], reverse=True
            )
            # keep the highest logit as order, set others to next candidate
            for idx, _ in idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()

    return ret


def check_duplicate(a: List[int]) -> bool:
    return len(a) != len(set(a))

def scale_bbox(bbox, width, height):
    x_scale = 1000.0 / width
    y_scale = 1000.0 / height

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
def layout_reader(bboxes, model_path=None, page_img_file=None):
    img = None
    scaled_bbox = bboxes
    if model_path is None:
        model_path = "./models/LayoutReader"
    if page_img_file is not None:
        img = cv2.imread(page_img_file)
        scaled_bbox = scale_bbox(bboxes, img.shape[1], img.shape[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        LayoutLMv3ForTokenClassification.from_pretrained(
            model_path
        )
            .bfloat16()
            .to(device)
            .eval()
    )
    inputs = boxes2inputs(scaled_bbox)
    inputs = prepare_inputs(inputs, model)
    logits = model(**inputs).logits.cpu().squeeze(0)
    result = parse_logits(logits, len(scaled_bbox))

    if img is not None:
        for idx, box in enumerate(bboxes):
            x0, y0, x1, y1 = box
            x0 = round(x0)
            y0 = round(y0)
            x1 = round(x1)
            y1 = round(y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 1)

            # 增大字体大小和线宽
            font_scale = 1.0  # 原先是0.5
            thickness = 2  # 原先是1

            cv2.putText(
                img,
                str(idx),
                (x1, y1),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale,
                (0, 0, 255),
                thickness,
            )
        cv2.imwrite(page_img_file, img)
    return result

if __name__ == '__main__':
    bboxes = [[584, 0, 595, 1], [35, 120, 89, 133],
              [35, 140, 75, 152]]
    result = layout_reader(bboxes)
    print(result)
