from paddleocr import PaddleOCR, draw_ocr
import cv2
import cv2
import requests
import numpy as np
from urllib.parse import urlparse
from io import BytesIO

# 图片前处理 灰度化+去噪
def _img_pretreatment(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoising = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    return denoising

# 读取图片
def _read_image(image_path):
    # 检测图像路径是本地路径还是 URL
    prefix_list = ['https', 'http', 'ftp']
    parsed_url = urlparse(image_path)
    prefix = parsed_url.scheme  # 如果没有 scheme，则视为本地路径
    if prefix not in prefix_list :
        image = cv2.imread(image_path)
    else:
        # 从 URL 地址读取图像
        response = requests.get(image_path)
        image_bytes = BytesIO(response.content)
        image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

# 识别图片
def _img_ocr(image, lang):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    ocr_result = []
    result = ocr.ocr(image, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            ocr_result.append([line[0],line[1][0]])
    return ocr_result

def image_2_txt(image_path, lang='ch'):
    image = _read_image(image_path)
    pretreat_img = _img_pretreatment(image)
    txt_result = _img_ocr(pretreat_img, lang)
    return txt_result

if __name__ == "__main__":
    import time
    strat = time.time()
    # image_path = "D:\\fandow_project\\ppocr\\img_data\\1.png"
    image_path = "https://img-blog.csdnimg.cn/20201029092421379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDUxMTgw,size_16,color_FFFFFF,t_70#pic_center"
    # test = image_2_txt(image_path)
    # for i in test:
    #     print(i)
    import multiprocessing
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # 并行处理单张图片
    result = pool.apply(image_2_txt, args=(image_path,))
    for i in result:
        print(result)

    end = time.time()
    print(f"cost: {end-strat}")

    
