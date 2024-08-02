import json
import os

import cv2
import numpy as np
import yaml
from PIL import Image
from paddleocr import PPStructure

from modules.extract_pdf import load_pdf_fitz, load_pdf_fitz_with_img_return
from modules.layoutReader.img_result_render import layout_box_order_render_with_label
from modules.layoutReader.layout_sort_rules import sorted_layout_boxes
from modules.post_process import pe_res_trans_2_layout_box, filter_consecutive_boxes, layout_abandon_fix_to_text, \
    header_footer_fix_by_paddleocr, layout_box_trans_2_pe_res, get_croped_image
from pdf_extract import layout_model_init, tr_model_init, mfr_model_init
from torchvision import transforms
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'


with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)
device = model_configs['model_args']['device']
dpi = model_configs['model_args']['pdf_dpi']

# tr_model = tr_model_init(model_configs['model_args']['tr_weight'], max_time=model_configs['model_args']['table_max_time'], device=device)
mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
mfr_transform = transforms.Compose([mfr_vis_processors, ])
layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
layout_reader_path = model_configs['model_paths']['layout_reader_path']
paddle_layout_engine = PPStructure(table=False, ocr=False, show_log=True)


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
    img_list, img_np_list = load_pdf_fitz_with_img_return(file_path, dpi=dpi)
    doc_layout_result = []

    for i, img in enumerate(img_list):
        # 保存原图
        img_save_path = f'{output_dir}/page{i}.jpg'
        img.save(f'{output_dir}/page{i}.jpg')
        bgr_array = img_np_list[i]
        label_boxes = single_page_img_parse(bgr_array, img, img_save_path, output_dir)
        # 存储为pe的json格式
        pe_res = layout_box_trans_2_pe_res(label_boxes)
        img_H, img_W = bgr_array.shape[0], bgr_array.shape[1]
        pe_res['page_info'] = dict(
            page_no=i,
            height=img.height,
            width=img_W
        )
        doc_layout_result.append(pe_res)
    json.dump(doc_layout_result, open(f'{output_dir}/{file_name_with_ext}.json', 'w'))
    # ocr_extract(doc_layout_result, img_list)


def parse_layout_from_img():
    image_output = f'./output/pe'
    os.makedirs(image_output, exist_ok=True)
    image_ori = f'./output/sorted/latex/page0.jpg'
    with Image.open(image_ori) as img:
        # 将PIL图像转换为RGB模式的NumPy数组（PIL默认打开可能是RGB或RGBA，这取决于图像）
        img_array = np.array(img.convert('RGB'))
        # 转换颜色空间从RGB到BGR
        bgr_array = img_array[:, :, ::-1]
    single_page_img_parse(bgr_array, img, image_ori, image_output)


def single_page_img_parse(bgr_array, img, image_ori, image_output):
    layout_res = layout_model(bgr_array, ignore_catids=[], min_score=0.5)
    label_boxes = pe_res_trans_2_layout_box(layout_res)
    # 过滤重叠和覆盖的检测框
    label_boxes, valid_idx = filter_consecutive_boxes(label_boxes)
    # 利用paddle解决错误识别为其他类型的header和footer
    label_boxes = header_footer_fix_by_paddleocr(bgr_array, label_boxes, ppstructure=paddle_layout_engine)
    # 排序检测框
    label_boxes = sorted_layout_boxes(label_boxes, img.width)
    # label_boxes, orders = layout_reader_sort(label_boxes, layout_reader_path, img.width, img.height)
    # 修正错误的页眉页脚识别为plain_text
    label_boxes, _ = layout_abandon_fix_to_text(label_boxes)
    # 去除所有页眉页脚
    label_boxes = [box for _, box in enumerate(label_boxes) if box.label != 'abandon']
    # 按顺序渲染
    layout_box_order_render_with_label(label_boxes, image_ori, image_output)
    return label_boxes


if __name__ == '__main__':
    # parse_from_img()
    # ocr_extract()
    pdf_layout_parse()
    # print(layout_res)
