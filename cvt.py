from math import ceil

import numpy as np
import cv2

# 1byte的长宽信息
META_W = 2
META_H = 4

BASE_ORD = 10240


def im_read(path, flags=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def fix_you(origin_length, time_a, time_b):
    return ceil(origin_length / (time_a * time_b)) * time_a * time_b


# 这里怎么出的错
def get_fix_w_h(origin_w, origin_h, col_x):
    # 修正宽
    fixed_w = fix_you(origin_w, col_x, META_W)
    fixed_h = ceil(fixed_w * origin_h / origin_w / META_H) * META_H

    return fixed_w, fixed_h


def fix_img(org_img, col_x):
    org_h, org_w = org_img.shape
    fixed_w, fixed_h = get_fix_w_h(org_w, org_h, col_x)
    return cv2.resize(org_img, (fixed_w, fixed_h))


def init_img(image: np.ndarray, col_x) -> np.ndarray:
    """
    初始化图片，包括灰度化，resize成合适长宽

    :param image: 图片
    :param col_x: 文本宽，列数
    :return:
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return fix_img(image, col_x)


def array2bin(array, upper_bound, threshold: float):
    """
    二值化

    :param threshold: 归一化阈值[0,1]之间
    :param upper_bound: 上界值，一般是255
    :param array: 要二值化的数组
    :return:
    """
    return (np.sum(array) / (np.prod(array.shape) * upper_bound)) >= threshold


# 位置和位的映射，例如第一行第一个是第一位，第二行第一个是第二位，第一行第二个是第四位
position_map = {
    (0, 0): 0,
    (1, 0): 1,
    (2, 0): 2,
    (3, 0): 3,
    (0, 1): 4,
    (1, 1): 5,
    (2, 1): 6,
    (3, 1): 7,
}


def array2char(array: np.ndarray, no_black):
    """
    数组转换成字符

    :param array:
    :param no_black:
    :return:
    """
    o = BASE_ORD
    for position, digit in position_map.items():
        o += array[position] * (2 ** digit)
    # 纯黑替换
    if no_black and o == BASE_ORD:
        o += 2 ** position_map[3, 1]
    return chr(o)


def image_shrink_and_bin(image_resized, col_x, upper_bound, threshold: float):
    h, w = image_resized.shape
    shrink_w = col_x * META_W
    shrink_h = shrink_w * h // w
    new_img = np.zeros((shrink_h, shrink_w), dtype=bool)
    # 大像素点宽高
    pixel_w = w // shrink_w
    pixel_h = h // shrink_h
    for x in range(shrink_w):
        for y in range(shrink_h):
            arr = image_resized[pixel_h * y:pixel_h * (y + 1), pixel_w * x:pixel_w * (x + 1)]
            new_img[y, x] = array2bin(arr, upper_bound, threshold)
    return new_img


def shrunk_image_to_txt_array(shrunk_image: np.ndarray, no_black):
    shrink_h, shrink_w = shrunk_image.shape
    row_x = shrink_h // META_H
    col_x = shrink_w // META_W
    txt_arr = np.zeros((row_x, col_x), dtype=object)
    for x in range(col_x):
        for y in range(row_x):
            arr = shrunk_image[META_H * y:META_H * (y + 1), META_W * x:META_W * (x + 1)]
            txt_arr[y, x] = array2char(arr, no_black)
    return txt_arr


def image2txt(image, col_x=20, threshold=0.5, no_black=True, image_inverse=False):
    image = init_img(image, col_x)
    shrunk_bin_image = image_shrink_and_bin(image, col_x, 255, threshold)
    if image_inverse:
        shrunk_bin_image = ~shrunk_bin_image
    txt_array = shrunk_image_to_txt_array(shrunk_bin_image, no_black)
    txt = '\n'.join(
        ''.join(
            char for char in line
        )
        for line in txt_array
    )
    return txt


if __name__ == '__main__':
    print(image2txt(im_read(r".\image\sun.jpg"), col_x=50, threshold=0.5, no_black=True, image_inverse=False))
