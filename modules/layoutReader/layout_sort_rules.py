import cv2
import numpy as np

from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox


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


if __name__ == '__main__':
    bboxes = [[584, 0, 595, 1], [35, 120, 89, 133],
              [35, 140, 75, 152]]
    sorted_layout_boxes(bboxes, 1000)
