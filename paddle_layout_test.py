import json
import os

import numpy as np
import yaml
from PIL import Image
from paddleocr import PPStructure
from utils import LayoutBox, trans_2_layout_box, filter_consecutive_boxes, sorted_layout_boxes, \
    layout_box_trans_2_pe_res
from modules.layoutReader.img_result_render import layout_box_order_render_with_label
# from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox
# from modules.post_process import filter_consecutive_boxes, pe_res_trans_2_layout_box, trans_2_layout_box
# from modules.layoutReader.layout_sort_rules import sorted_layout_boxes
from modules.extract_pdf import load_pdf_fitz, load_pdf_fitz_with_img_return

# with open('configs/model_configs.yaml') as f:
#     model_configs = yaml.load(f, Loader=yaml.FullLoader)
# device = model_configs['model_args']['device']
# dpi = model_configs['model_args']['pdf_dpi']


table_engine = PPStructure(recovery=True, table=True, ocr=True, show_log=True)


def pdf_layout_parse():
    file_path = f'./pdfs/latex.pdf'
    # 保存本页图片信息
    # 提取文件名（包括扩展名）
    file_name_with_ext = os.path.basename(file_path)
    # 移除扩展名
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
    output_dir = f'output/sorted/{file_name_without_ext}'
    os.makedirs(output_dir, exist_ok=True)
    # 将pdf读取为图片+图片np数组
    img_list, img_np_list = load_pdf_fitz_with_img_return(file_path, dpi=200)
    doc_layout_result = []

    for i, img in enumerate(img_list):
        # 保存原图
        img_save_path = f'{output_dir}/page{i}.jpg'
        img.save(f'{output_dir}/page{i}.jpg')
        bgr_array = img_np_list[i]
        paddle_res,label_boxes = single_page_parse(bgr_array, img, img_save_path, output_dir)
        for res in paddle_res:
            res['page_info'] = dict(
                page_no=i,
                height=img.height,
                width=img.width
            )
            res['img'] = None

        doc_layout_result.append(paddle_res)
        # 存储为pe的json格式
        # img_H, img_W = bgr_array.shape[0], bgr_array.shape[1]
        # label_boxes =[label_box.__dict__ for label_box in label_boxes]
        # for res in paddle_res
        # paddle_res_store = {
        #     'det_boxes':[res in paddle_res]
        #     "box": label_boxes,
        #     "page_info": dict(
        #         page_no=i,
        #         height=img.height,
        #         width=img_W
        #     )
        # }
        # doc_layout_result.append(paddle_res)
    json.dump(doc_layout_result, open(f'{output_dir}/{file_name_with_ext}.json', 'w'))

    # ocr_extract(doc_layout_result, img_list)


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


def single_page_parse(bgr_array, img, img_save_path, output_dir):
    paddle_layout_res = table_engine(bgr_array)
    label_bboxes = trans_2_layout_box(paddle_layout_res)
    # 过滤重叠和覆盖的检测框
    label_bboxes, valid_idx = filter_consecutive_boxes(label_bboxes)
    # 去除所有页眉页脚
    label_bboxes = [box for _, box in enumerate(label_bboxes) if box.label != 'header' and box.label != 'footer']
    # 排序检测框
    label_bboxes = sorted_layout_boxes(label_bboxes, img.width)
    # label_bboxes, orders = layout_reader_sort(label_bboxes, layout_reader_path, img.width, img.height)
    layout_box_order_render_with_label(label_bboxes, img_save_path, output_dir)
    return paddle_layout_res,label_bboxes


def parse_from_img():
    global model_configs
    output_dir = f'./output/paddle'
    img_save_path = f'./output/sorted/latex/page1.jpg'
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    with Image.open(img_save_path) as img:
        # 将PIL图像转换为RGB模式的NumPy数组（PIL默认打开可能是RGB或RGBA，这取决于图像）
        img_array = np.array(img.convert('RGB'))
        # 转换颜色空间从RGB到BGR
        bgr_array = img_array[:, :, ::-1]
    single_page_parse(bgr_array, img, img_save_path, output_dir)


if __name__ == '__main__':
    parse_from_img()
    # pdf_layout_parse()
