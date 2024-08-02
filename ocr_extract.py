import argparse
import json
import os
import time

import cv2
import torch
import yaml
from PIL import Image
from struct_eqtable import build_model

from modules.extract_pdf import load_pdf_fitz_with_img_return
from modules.layoutlmv3.layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutBox, formula_types
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.post_process import get_croped_image, pe_res_trans_2_layout_box, latex_rm_whitespace
from modules.self_modify import ModifiedPaddleOCR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
# from pdf_extract import tr_model_init, mfr_model_init, MathDataset
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor


class MathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]
        if self.transform:
            image = self.transform(raw_image)
        return image


def mfr_model_init(weight_dir, device='cpu'):
    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)
    cfg = Config(args)
    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model = model.to(device)
    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)
    return model, vis_processor

def layout_model_init(weight):
    model = Layoutlmv3_Predictor(weight)
    return model

def tr_model_init(weight, max_time, device='cuda'):
    tr_model = build_model(weight, max_new_tokens=4096, max_time=max_time)
    if device == 'cuda':
        tr_model = tr_model.cuda()
    return tr_model




continuealbe_labels = ['plain_text', 'table']
ocr_model = ModifiedPaddleOCR(show_log=True)
with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)
device = model_configs['model_args']['device']
dpi = model_configs['model_args']['pdf_dpi']
tr_model = tr_model_init(model_configs['model_args']['tr_weight'],
                         max_time=model_configs['model_args']['table_max_time'], device=device)
mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
mfr_transform = transforms.Compose([mfr_vis_processors, ])



def get_merge_box_crop_img(img, label_box):
    single_imgs = []
    for single_box_area in label_box.merged_bbox:
        xmin, ymin, xmax, ymax = single_box_area
        single_bbox_img = get_croped_image(img, [xmin, ymin, xmax, ymax])
        single_imgs.append(single_bbox_img)
    # 定义边缘空白
    edge_margin = 50
    # 计算最大宽度和累计高度
    max_width = max(img.width for img in single_imgs)
    crop_img_total_height = sum(img.height for img in single_imgs)  # 加上上下边缘的空白
    # 创建一个新的空白图像
    bbox_img = Image.new('RGB',
                         (max_width + 2 * edge_margin, crop_img_total_height + 2 * edge_margin),
                         'white')
    # 计算每个图像的粘贴位置
    current_y = edge_margin  # 开始位置在顶部空白下方
    for img in single_imgs:
        bbox_img.paste(img, (edge_margin, current_y))  # 粘贴图像
        current_y += img.height  # 更新下一张图像的起始位置
    return img


def handle_page_between_box_merge(all_label_boxes):
    """
    将跨页的分块整合起来，可能出现某页为空
    """
    for i in range(len(all_label_boxes) - 1):
        if not all_label_boxes[i]:
            continue
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


def ocr_extract(file_path=None, doc_layout_result=None, img_list=None):
    if file_path is None:
        file_path = f'./pdfs/lunwen.pdf'
    file_name_with_ext = os.path.basename(file_path)
    # 移除扩展名
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
    output_dir = f'output/sorted/{file_name_without_ext}'

    if img_list is None:
        img_list, img_np_list = load_pdf_fitz_with_img_return(file_path, dpi=dpi)
    if doc_layout_result is None:
        with open(f'{output_dir}/{file_name_with_ext}.json', 'r') as file:
            # 使用 json.load() 函数读取文件内容并解析为Python数据结构
            doc_layout_result = json.load(file)

    # 转换为layout_box结构
    all_label_boxes = [pe_res_trans_2_layout_box(page_res) for page_res in doc_layout_result]
    # 拼装同页的跨列数据
    all_label_boxes = [handle_page_inner_box_merge(label_boxes) for label_boxes in all_label_boxes]
    # 拼装跨页的数据
    all_label_boxes = handle_page_between_box_merge(all_label_boxes)
    # 进行ocr识别解析
    ocr_res_list = []
    # 每页对应的公式解析的latex结果
    page_formulas = ocr_formula_batch(all_label_boxes, img_list)

    for i, img in enumerate(img_list):
        label_boxes = all_label_boxes[i]
        if len(label_boxes) == 0:
            continue
        for j, label_box in enumerate(label_boxes):
            start = time.time()
            ocr_parse_res = {
                'label': label_box.label,
                'text': None,
                'time_spend': None,
                'page_num': i,
                'page_height': img.height,
                'page_width': img.width,
                'img': None,
                'latex': None,
                'bbox': label_box.bbox,
                'box_type': label_box.box_type,
                'merged_bbox': label_box.merged_bbox,
                'ocr_res': None,
            }
            if label_box.label in ['title', 'figure_caption', 'table_caption', 'table_footnote', 'plain_text']:
                ocr_res, text = ocr_rec_text(img, label_box)
                ocr_parse_res['text'] = text
            elif label_box.label == 'table':
                ocr_res, latex = ocr_rec_table(img, label_box)
                ocr_parse_res['latex'] = latex
            elif label_box.label == 'isolate_formula':
                ocr_parse_res['latex'] = page_formulas[i][j]
                ocr_res = None
            else:
                # 其它类型不进行识别，但是需要站位
                ocr_res = None
            ocr_parse_res['ocr_res'] = ocr_res
            end = time.time()
            ocr_parse_res['time_spend'] = end - start
            ocr_res_list.append(ocr_parse_res)
    json.dump(ocr_res_list, open(f'{output_dir}/{file_name_with_ext}-orc.json', 'w'))


def ocr_formula_batch(all_label_boxes, img_list)-> list[list[str]]:
    mf_image_list = []
    page_formula_num = []
    for i, img in enumerate(img_list):
        label_boxes = all_label_boxes[i]
        # 对应空页
        if len(label_boxes):
            formula_idx = []
        else:
            formula_idx = [i for i, box in enumerate(label_boxes) if box.label in formula_types]
        page_formula_num.append(len(formula_idx))
        for i in formula_idx:
            xmin, ymin, xmax, ymax = label_boxes[i].bbox
            bbox_img = get_croped_image(img, [xmin, ymin, xmax, ymax])
            bbox_img = cv2.cvtColor(np.asarray(bbox_img), cv2.COLOR_RGB2BGR)
            mf_image_list.append(bbox_img)
    a = time.time()
    dataset = MathDataset(mf_image_list, transform=mfr_transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=4)
    mfr_res = []
    for imgs in dataloader:
        imgs = imgs.to(device)
        output = mfr_model.generate({'image': imgs})
        mfr_res.extend(output['pred_str'])
    mfr_latex = [latex_rm_whitespace(latex) for latex in mfr_res]
    # 初始化一个空的二维列表来存储每页的公式
    page_formulas = []
    # 当前页的起始索引
    start_index = 0
    # 遍历每页的公式数量
    for num_formulas in page_formula_num:
        # 计算当前页结束的索引
        end_index = start_index + num_formulas
        # 从 mfr_latex 中提取当前页的公式，并添加到 page_formulas
        page_formulas.append(mfr_latex[start_index:end_index])
        # 更新起始索引为下一页的开始
        start_index = end_index
    b = time.time()
    print("formula nums:", len(mf_image_list), "mfr time:", round(b - a, 2))
    return page_formulas

def ocr_rec_table(img, label_box):
    if label_box.box_type == 'merge':
        bbox_img = get_merge_box_crop_img(img, label_box)
    else:
        xmin, ymin, xmax, ymax = label_box.bbox
        bbox_img = get_croped_image(img, [xmin, ymin, xmax, ymax])
    with torch.no_grad():
        ocr_res = tr_model(bbox_img)
    latex = ocr_res[0]
    return ocr_res, latex


def ocr_rec_text(img, label_box):
    if label_box.box_type == 'merge':
        bbox_img = get_merge_box_crop_img(img, label_box)
    else:
        xmin, ymin, xmax, ymax = label_box.bbox
        bbox_img = get_croped_image(img, [xmin, ymin, xmax, ymax])
    # 获取ocr识别结果
    cropped_img = cv2.cvtColor(np.asarray(bbox_img), cv2.COLOR_RGB2BGR)
    ocr_res = ocr_model.ocr(cropped_img)[0]
    if ocr_res:
        all_text = [box_ocr_res[1][0] for box_ocr_res in ocr_res]
        return ocr_res, all_text
    return None,None


if __name__ == '__main__':
    os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
    ocr_extract()
