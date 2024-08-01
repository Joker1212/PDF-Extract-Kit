import json
import os

import numpy as np
import yaml
from PIL import Image
from paddleocr import PPStructure

from modules.layoutReader.img_result_render import layout_box_order_render_with_label
from modules.layoutReader.layout_reader import layout_reader_sort
from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox
from modules.post_process import filter_consecutive_boxes, pe_res_trans_2_layout_box, trans_2_layout_box
from modules.layoutReader.layout_sort_rules import sorted_layout_boxes

def get_crop_img(img, box):
    # 使用PIL的crop方法根据box裁剪图片
    cropped_img = img.crop(box)

    # 将裁剪后的PIL图像转换为NumPy数组
    cropped_img_array = np.array(cropped_img.convert('RGB'))

    # 如果需要转换颜色空间从RGB到BGR
    return cropped_img_array[:, :, ::-1], cropped_img


def read_box_from_json(json_file, page_index, box_index) -> LayoutBox:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return pe_res_trans_2_layout_box(data[page_index])[box_index]


def xiajibaxiede():
    file_path = "pdfs/latex.pdf"
    # 提取文件名（包括扩展名）
    file_name_with_ext = os.path.basename(file_path)
    # 移除扩展名
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
    print(file_name_without_ext)
    image_ori = f'./output/sorted/page0.jpg'
    image_output = f'./output/paddle/crop'
    image_crop_output_name = 'crop.jpg'
    layout_box = read_box_from_json('./output/latex.json', 0, 0)
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    with Image.open(image_ori) as img:
        # 将PIL图像转换为RGB模式的NumPy数组（PIL默认打开可能是RGB或RGBA，这取决于图像）
        crop_img_array, crop_img = get_crop_img(img, layout_box.bbox)
    crop_img.save(f'{image_output}/{image_crop_output_name}')
    layout_reader_path = model_configs['model_paths']['layout_reader_path']
    table_engine = PPStructure(table=False, ocr=False, show_log=True)
    paddle_layout_res = table_engine(crop_img_array)
    label_bboxes = trans_2_layout_box(paddle_layout_res)
    layout_box_order_render_with_label(label_bboxes, image_ori, image_output)


if __name__ == '__main__':
    image_output = f'./output/paddle'
    image_ori = f'./output/sorted/latex/page0.jpg'
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
    # 过滤重叠和覆盖的检测框
    label_bboxes, valid_idx = filter_consecutive_boxes(label_bboxes)
    # 排序检测框
    label_bboxes = sorted_layout_boxes(label_bboxes, img.width)

    # 排序检测框
    # label_bboxes, orders = layout_reader_sort(label_bboxes, layout_reader_path, img.width, img.height)

    layout_box_order_render_with_label(label_bboxes, image_ori, image_output)
