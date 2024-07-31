import numpy as np
import yaml
from PIL import Image

from modules.layoutReader.img_result_render import layout_box_order_render_with_label
from modules.layoutReader.layout_reader import layout_reader_sort
from modules.post_process import pe_res_trans_2_layout_box, filter_consecutive_boxes
from pdf_extract import layout_model_init

if __name__ == '__main__':
    image_output = f'./output/pe'
    image_ori = f'./output/sorted/page0.jpg'
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    with Image.open(image_ori) as img:
        # 将PIL图像转换为RGB模式的NumPy数组（PIL默认打开可能是RGB或RGBA，这取决于图像）
        img_array = np.array(img.convert('RGB'))
        # 转换颜色空间从RGB到BGR
        bgr_array = img_array[:, :, ::-1]

    layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
    layout_reader_path = model_configs['model_paths']['layout_reader_path']
    layout_res = layout_model(bgr_array, ignore_catids=[], min_score=0.5)
    label_bboxes = pe_res_trans_2_layout_box(layout_res)
    # 排序检测框
    label_bboxes, orders = layout_reader_sort(label_bboxes, layout_reader_path, img.width, img.height)
    # 过滤重叠和覆盖的检测框
    label_bboxes, valid_idx = filter_consecutive_boxes(label_bboxes)
    layout_box_order_render_with_label(label_bboxes, image_ori, image_output)
    print(layout_res)
