import json
import os

import cv2


def layout_box_order_render_with_label(layout_boxes, page_img_file, output_path=None):
    img = cv2.imread(page_img_file)
    # 检查图像是否成功读取
    if img is None:
        raise IOError(f"Failed to load image file: {page_img_file}")

    for idx, box in enumerate(layout_boxes):
        x0, y0, x1, y1 = box.bbox
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
            str(idx) + '.' + box.label,
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
