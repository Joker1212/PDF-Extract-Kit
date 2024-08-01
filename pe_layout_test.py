import json
import os

import numpy as np
import yaml
from PIL import Image
from paddleocr import PPStructure

from modules.extract_pdf import load_pdf_fitz, load_pdf_fitz_with_img_return
from modules.layoutReader.img_result_render import layout_box_order_render_with_label
from modules.layoutReader.layout_sort_rules import sorted_layout_boxes
from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox
from modules.post_process import pe_res_trans_2_layout_box, filter_consecutive_boxes, layout_abandon_fix_to_text, \
    header_footer_fix_by_paddleocr, layout_box_trans_2_pe_res
from modules.self_modify import ModifiedPaddleOCR
from pdf_extract import layout_model_init, tr_model_init

with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)
device = model_configs['model_args']['device']
dpi = model_configs['model_args']['pdf_dpi']

# tr_model = tr_model_init(model_configs['model_args']['tr_weight'], max_time=model_configs['model_args']['table_max_time'], device=device)
layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
ocr_model = ModifiedPaddleOCR(show_log=True)
layout_reader_path = model_configs['model_paths']['layout_reader_path']
table_engine = PPStructure(table=False, ocr=False, show_log=True)
continuealbe_labels = ['plain_text', 'table']

def pdf_parse():
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
    # 转换为layout_box结构
    all_label_boxes = [pe_res_trans_2_layout_box(page_res) for page_res in doc_layout_result]
    # 拼装同页的跨列数据
    all_label_boxes = [handle_page_inner_box_merge(label_boxes) for label_boxes in all_label_boxes]
    # 拼装跨页的数据
    all_label_boxes = handle_page_between_box_merge(all_label_boxes)
    #进行ocr识别解析
    # for label_boxes in all_label_boxes:

def handle_page_between_box_merge(all_label_boxes):
    for i in range(len(all_label_boxes) - 1):
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
                merge_label_box = next.merged_bbox
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


def parse_from_img():
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
    label_boxes = header_footer_fix_by_paddleocr(bgr_array, label_boxes, ppstructure=table_engine)
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
    parse_from_img()
    # pdf_parse()
    # print(layout_res)
