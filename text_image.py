from functools import partial
import warnings
from pathlib import Path
import struct

import numpy as np
import cv2

# 1byte的长宽信息
META_W = 2
META_H = 4

# 起始字符序号
BASE_ORD = 10240

# 空格offset
BLANK_OFFSET = ord(' ') - BASE_ORD

# 换行符
LINE_BREAK = '\n'

# 位置和位的映射，例如第一行第一个是第一位，第二行第一个是第二位，第一行第二个是第四位
POSITION_MAP = {
    (0, 0): 0,
    (1, 0): 1,
    (2, 0): 2,
    (3, 0): 3,
    (0, 1): 4,
    (1, 1): 5,
    (2, 1): 6,
    (3, 1): 7,
}
# 储存每个符号用的类型
PIXEL_TYPE = np.uint8
PIXEL_BYTE = 1

REPLACE_BLACK_PIXEL = 2 ** POSITION_MAP[3, 1]

# 转换矩阵
TRANSFORM_MATRIX = np.empty(shape=(META_H, META_W), dtype=PIXEL_TYPE)

for position, index in POSITION_MAP.items():
    TRANSFORM_MATRIX[position] = 2 ** index


def array2int(array: np.ndarray) -> int:
    return (array * TRANSFORM_MATRIX).sum()


def int2char(pixel_int: int, blank_offset=BLANK_OFFSET) -> str:
    return chr(
        BASE_ORD + (blank_offset if pixel_int == 0 else pixel_int)
    )


def sure_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def im_read(file, flags=cv2.IMREAD_GRAYSCALE):
    """
    中文路径和filestorage
    """
    return cv2.imdecode(np.fromfile(file, dtype=np.uint8), flags)


class TextImage:
    SUFFIX = '.ti'
    BIN_HEAD = b'TI'
    ATTR_AND_FORMAT = [('w', '<I'), ('h', '<I')]
    ATTR_INDEX = {
        attr: index
        for index, (attr, _) in enumerate(ATTR_AND_FORMAT)
    }

    def __init__(self, array: np.ndarray, w=None, h=None, blank_offset=BLANK_OFFSET):  # 准备用numpy存数据
        assert w != 0 and h != 0, '宽高不能为0'
        if w and h:
            # reshape一下
            array.shape = (h, w)
        else:
            assert len(array.shape) == 2, '不指定宽高则传入数组必须为二维的'

        self.data = array  # type: np.ndarray
        # h=字符行数, w=每行字符数
        self.h, self.w = self.data.shape
        self.blank_offset = blank_offset

    def to_str(self, blank_offset=BLANK_OFFSET) -> str:
        # 提前赋值
        map_func = partial(int2char, blank_offset=blank_offset)

        return LINE_BREAK.join(
            ''.join(map(map_func, line_arr))
            for line_arr in self.data
        )

    def __str__(self):
        return self.to_str(blank_offset=self.blank_offset)

    def to_bytes(self):
        return self.data.tobytes()

    def save(self, file):
        file = Path(file)
        if file.suffix.lower() != self.SUFFIX:
            warnings.warn(f"file suffix better be {self.SUFFIX}, but yours is {file.suffix}")
        with open(file, mode='wb') as fp:
            fp.write(self.BIN_HEAD)
            for attr, fmt in self.ATTR_AND_FORMAT:
                fp.write(struct.pack(fmt, self.__getattribute__(attr)))
            fp.write(self.to_bytes())

    @classmethod
    def from_text_image_file(cls, ti_file, blank_offset=BLANK_OFFSET):
        with open(ti_file, mode='rb') as fp:
            this_bin_head = fp.read(len(cls.BIN_HEAD))
            assert this_bin_head == cls.BIN_HEAD, f'bin head should be {cls.BIN_HEAD}, but yours is {this_bin_head}'
            info_list = [
                # because unpack return a tuple
                struct.unpack(fmt, fp.read(struct.calcsize(fmt)))[0]
                for attr, fmt in cls.ATTR_AND_FORMAT
            ]
            w = info_list[cls.ATTR_INDEX['w']]
            h = info_list[cls.ATTR_INDEX['h']]
            array = np.frombuffer(fp.read(), dtype=PIXEL_TYPE)  # type:np.ndarray
            assert array.size == w * h, f'file is just not right, expect w * h = {w * h}, but read {array.size}'
            return cls(array=array, w=w, h=h, blank_offset=blank_offset)

    @classmethod
    def from_buffer(cls, buffer, w: int, h: int):
        return cls(np.frombuffer(buffer, dtype=PIXEL_TYPE), w=w, h=h)

    @classmethod
    def from_image(cls, image: np.ndarray, n, threshold=127, image_inverse=False, blank_offset=BLANK_OFFSET):
        """
        图片转字符串

        :param image: 灰度图，不灰度也给你变灰度
        :param n: 一行几个字符
        :param threshold: 二值化阈值
        :param image_inverse: 是否反相
        :param blank_offset: 空白字符序号的偏移量
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

        # 文本图片的宽
        c_w = n
        # 文本图片的高
        c_h = n_h // META_H
        return cls(
            np.fromiter(
                (
                    array2int(mask[META_H * y:META_H * (y + 1), META_W * x:META_W * (x + 1)])
                    for y in range(c_h)  # 文本行数
                    for x in range(c_w)  # 每行字符数
                ),
                dtype=PIXEL_TYPE
            ),
            w=c_w,
            h=c_h,
            blank_offset=blank_offset
        )


def show(file, n, threshold, image_inverse, replace_blank):
    offset = BLANK_OFFSET if replace_blank else REPLACE_BLACK_PIXEL
    file = Path(file)
    if file.suffix.lower() == TextImage.SUFFIX:
        ti = TextImage.from_text_image_file(file, blank_offset=offset)
    else:
        ti = TextImage.from_image(im_read(str(file)), n, threshold, image_inverse, blank_offset=offset)
    print(ti)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("文本图片显示")
    parser.add_argument('file', type=str, help='图片地址')
    parser.add_argument('-n', type=int, help='每行字符数', default=50)
    parser.add_argument('--threshold', type=int, help="二值化阈值，0到255", default=127)
    parser.add_argument('--image_inverse', action="store_true", help="图像反相，黑白颠倒")
    parser.add_argument('--replace_blank', action="store_false", help="空白替换")

    args = parser.parse_args().__dict__

    show(**args)
