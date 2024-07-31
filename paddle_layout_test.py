import numpy as np
import yaml
from PIL import Image
from paddleocr import PPStructure

from modules.layoutReader.img_result_render import layout_box_order_render_with_label
from modules.layoutReader.layout_reader import layout_reader_sort
from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox
from modules.post_process import filter_consecutive_boxes


def trans_2_layout_box(paddle_layout_res):
    label_bboxes = []
    for idx, paddle_dict in enumerate(paddle_layout_res):
        bbox = LayoutBox(paddle_dict['bbox'], paddle_dict['type'])
        label_bboxes.append(bbox)
    return label_bboxes

if __name__ == '__main__':
    image_output = f'./output/paddle'
    image_ori = f'./output/sorted/page1.jpg'
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    with Image.open(image_ori) as img:
        # 将PIL图像转换为RGB模式的NumPy数组（PIL默认打开可能是RGB或RGBA，这取决于图像）
        img_array = np.array(img.convert('RGB'))
        # 转换颜色空间从RGB到BGR
        bgr_array = img_array[:, :, ::-1]

    layout_reader_path = model_configs['model_paths']['layout_reader_path']
    table_engine = PPStructure(table=False, ocr=False, show_log=True)
    paddle_layout_res = table_engine(bgr_array)

    label_bboxes = trans_2_layout_box(paddle_layout_res)
    # 排序检测框
    label_bboxes, orders = layout_reader_sort(label_bboxes, layout_reader_path, img.width, img.height)
    # 过滤重叠和覆盖的检测框
    label_bboxes, valid_idx = filter_consecutive_boxes(label_bboxes)
    layout_box_order_render_with_label(label_bboxes, image_ori, image_output)

