import numpy as np
import cv2

# 1byte的长宽信息
META_W = 2
META_H = 4

BASE_ORD = 10240

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


def sure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def array2char(array: np.ndarray):
    """
    数组转换成字符
    """
    o = BASE_ORD
    for position, digit in position_map.items():
        o += array[position] * (2 ** digit)
    # 纯黑替换成右下角的点
    if o == BASE_ORD:
        o += 2 ** position_map[3, 1]
    return chr(o)


def im_read(file, flags=cv2.IMREAD_GRAYSCALE):
    """
    中文路径和filestorage
    """
    return cv2.imdecode(np.fromfile(file, dtype=np.uint8), flags)


def image2text(image: np.ndarray, n, threshold=127, image_inverse=False):
    """
    图片转字符串

    :param image: 二值化图片
    :param n: 一行几个字符
    :param threshold: 二值化阈值
    :param image_inverse: 是否反相
    :return:
    """
    # 至少一个单位行吧
    assert n >= 1
    image = sure_gray(image)
    o_h, o_w = image.shape
    # 新宽
    n_w = n * META_W
    # 新高度，至少为1个单位高度
    n_h = max(
        META_H,
        round(o_h * n_w / o_w / META_H) * META_H
    )
    new_image = cv2.resize(image, (n_w, n_h), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(new_image, threshold, 255, cv2.THRESH_BINARY)
    mask = mask == 255
    if image_inverse:
        mask = ~mask
    text = '\n'.join(
        ''.join(
            array2char(mask[META_H * y:META_H * (y + 1), META_W * x:META_W * (x + 1)])
            for x in range(n)  # 每行字符数
        )
        for y in range(n_h // META_H)  # 文本行数
    )
    return text


if __name__ == '__main__':
    print(image2text(
        im_read(r"image/sun.jpg"),
        50,
        image_inverse=False
    ))
