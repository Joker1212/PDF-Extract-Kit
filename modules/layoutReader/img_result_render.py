import json
import os

import cv2

from modules.post_process import filter_consecutive_boxes


def layout_sorted_img_render(bboxes, page_img_file, output_path=None):
    img = cv2.imread(page_img_file)
    # 检查图像是否成功读取
    if img is None:
        raise IOError(f"Failed to load image file: {page_img_file}")

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

        # 修改文件名，添加_output后缀
    base_name = os.path.splitext(os.path.basename(page_img_file))[0]
    ext = os.path.splitext(page_img_file)[1]
    new_file_name = f"{base_name}_output{ext}"
    if output_path is not None:
        save_path = os.path.join(output_path, new_file_name)
    else:
        save_path = os.path.join(os.path.dirname(page_img_file), new_file_name)

    cv2.imwrite(save_path, img)

if __name__ == '__main__':
    img_path = f'../../output/sorted/page1.jpg'
    json_path = f"../../output/latex.json"
    with open(json_path, 'r', encoding='utf-8') as file:
        # 使用json.load()函数读取文件内容并转换为Python数据结构（通常是字典或列表）
        data = json.load(file)
        page = data[1]
        single_page_res = page["layout_dets"]
        bounding_boxes = []
        idx = 0
        for res in single_page_res:
            # if idx == 10 or idx == 11:
            if True:
                xmin, ymin = int(res['poly'][0]), int(res['poly'][1])
                xmax, ymax = int(res['poly'][4]), int(res['poly'][5])
                # 将坐标值组合成一个四元组，并添加到列表中
                bounding_boxes.append((xmin, ymin, xmax, ymax))
            idx += 1
        bounding_boxes = filter_consecutive_boxes(bounding_boxes)
        # bounding_boxes = layout_sorted_img_render(bounding_boxes, img_path)
