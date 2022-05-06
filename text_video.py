import curses
import datetime
import os
import struct
import time
import warnings
from math import ceil
from pathlib import Path
from typing import List, Optional
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm

from text_image import TextImage, LINE_BREAK, BASE_ORD, PIXEL_BYTE, PIXEL_TYPE

# 默认编码方式
DEFAULT_ENCODING = 'utf-8'


def image2text_for_pool(frame_arg):
    """
    多进程用

    :param frame_arg:
    :return:
    """
    frame, n, threshold, image_inverse = frame_arg
    return TextImage.from_image(frame, n, threshold, image_inverse)


class TextVideo:
    # 文本文件后缀
    SUFFIX = '.tv'
    # 渲染文件中，参数存储顺序
    ATTR_AND_FORMAT = [('fps', '<f'), ('w', '<I'), ('h', '<I'), ('frame_number', '<I')]
    # INDEX_ATTR = 'fps w h frame_number'.split()
    ATTR_INDEX = {
        attr: index
        for index, (attr, _) in enumerate(ATTR_AND_FORMAT)
    }
    SEPARATOR = ' '
    BIN_HEAD = b'TV'

    def __init__(self, image_list: List[TextImage], fps):
        self.frames = image_list
        self.w = self.frames[0].w
        self.h = self.frames[0].h
        self.fps = fps
        self.T = 1 / fps
        self.total_time = self.T * self.frame_number

    @property
    def frame_number(self):
        return len(self.frames)

    def get_image(self, delta_t):
        try:
            return self.frames[int(delta_t / self.T)]
        except IndexError:
            return None

    def save(self, file):
        file = Path(file)
        if file.suffix.lower() != self.SUFFIX:
            warnings.warn(f"file suffix better be {self.SUFFIX}, but yours is {file.suffix}")
        with open(file, mode='wb') as fp:
            fp.write(self.BIN_HEAD)
            for attr, fmt in self.ATTR_AND_FORMAT:
                fp.write(struct.pack(fmt, self.__getattribute__(attr)))
            for ti in self.frames:
                fp.write(ti.to_bytes())

        # with open(file, mode='wt', encoding=DEFAULT_ENCODING) as fp:
        #     # 写入基础信息
        #     fp.write(self.SEPARATOR.join(
        #         str(self.__getattribute__(attr))
        #         for attr in self.INDEX_ATTR
        #     ))
        #     # 写入每一帧
        #     for image in self.frames:
        #         fp.write(LINE_BREAK)
        #         fp.write(str(image))

    @classmethod
    def from_real_video_file(cls, real_video_file, n, threshold=127, image_inverse=False):
        # 视频流读取对象
        video_capture = cv2.VideoCapture(str(real_video_file))
        # 帧率
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        # 总帧数，用来显示读取进度
        frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # 进程数量，别用完cpu，留一个，省了假死
        processes = max(1, os.cpu_count() - 1)
        # 每一帧，加上转换的参数，整成一个生成器
        frame_arg_gen = (
            (video_capture.read()[1], n, threshold, image_inverse)
            for i in range(frame_number)
        )
        # 进程池
        with multiprocessing.Pool(processes=processes) as pool:
            text_image_list = list(
                # 加个进度条
                tqdm(pool.imap(image2text_for_pool, frame_arg_gen), total=frame_number, desc='渲染中'))

        video_capture.release()
        return cls(
            image_list=text_image_list,
            fps=fps
        )

    @classmethod
    def read_info(cls, tv_file):
        with open(tv_file, mode='rb') as fp:
            return cls._read_info_from_fp(fp)

    @classmethod
    def _read_info_from_fp(cls, fp):
        this_bin_head = fp.read(len(cls.BIN_HEAD))
        assert this_bin_head == cls.BIN_HEAD, f'bin head should be {cls.BIN_HEAD}, but yours is {this_bin_head}'
        return [
            struct.unpack(fmt, fp.read(struct.calcsize(fmt)))[0]
            for attr, fmt in cls.ATTR_AND_FORMAT
        ]

    @classmethod
    def from_text_video_file(cls, tv_file):
        with open(tv_file, mode='rb') as fp:
            info_list = cls._read_info_from_fp(fp)
            fps = info_list[cls.ATTR_INDEX['fps']]
            w = info_list[cls.ATTR_INDEX['w']]
            h = info_list[cls.ATTR_INDEX['h']]
            frame_number = info_list[cls.ATTR_INDEX['frame_number']]
            a_frame_length = w * h * PIXEL_BYTE
            text_image_list = [
                TextImage(
                    array=np.frombuffer(fp.read(a_frame_length), dtype=PIXEL_TYPE),
                    w=w,
                    h=h
                )
                for i in range(frame_number)
            ]
            return cls(
                image_list=text_image_list,
                fps=fps
            )

        # with open(text_video_file, mode='rt', encoding=DEFAULT_ENCODING) as fp:
        #     # 读取基础信息，去掉最后的换行符，用分隔符分割
        #     info_list = fp.readline().replace(LINE_BREAK, '').split(cls.SEPARATOR)
        #     fps = float(info_list[cls.ATTR_INDEX['fps']])
        #     w = int(info_list[cls.ATTR_INDEX['w']])
        #     h = int(info_list[cls.ATTR_INDEX['h']])
        #     frame_number = int(info_list[cls.ATTR_INDEX['frame_number']])
        #     # 将剩余文本读取入图像队列
        #     text_image_list = [
        #         TextImage([
        #             # 读每一行，取宽度个字符，因为后面字符可能是换行用的
        #             fp.readline()[:w]
        #             for j in range(h)
        #         ], copy=False)
        #         for i in range(frame_number)
        #     ]
        #     return cls(text_image_list, fps)


class TextVideoPlayer:
    PROCESS_BAR_CHAR = chr(BASE_ORD + 255)

    def __init__(self):
        # 请求次数
        self.info = False
        self.video = None  # type: Optional[TextVideo]
        self.count = 0

    def _get_fps_top(self, delta_t):
        try:
            return self.count / delta_t
        except ZeroDivisionError:
            return float("inf")

    def _get_info_text(self, delta_t):
        info_list = [
            # fps
            f"fps: {self._get_fps_top(delta_t)}",
            # 时间信息
            f"{datetime.timedelta(seconds=delta_t)}/{datetime.timedelta(seconds=self.video.total_time)}",
            # 进度条
            self.PROCESS_BAR_CHAR * int(self.video.w * delta_t / self.video.total_time)
        ]
        return LINE_BREAK.join(info_list)

    def play(self, video, info):
        self.count = 0
        self.video = video
        self.info = info
        curses.wrapper(self._play_process)

    def _play_process(self, std_scr):
        # 开始播放时间
        t0 = time.time()
        while True:
            std_scr.clear()
            self.count += 1
            # 经历的时间
            delta_t = time.time() - t0
            image = self.video.get_image(delta_t)
            if image is None:
                break
            # 显示字符图片
            std_scr.addstr(str(image))
            if self.info:
                # 显示详细信息
                std_scr.addstr(f'{LINE_BREAK}{self._get_info_text(delta_t)}')
            std_scr.refresh()
        # 播放结束
        std_scr.clear()
        std_scr.addstr("END")
        std_scr.refresh()
        std_scr.getch()


def play(file, n=50, threshold=127, image_inverse=False, info=False):
    file = Path(file)
    # 如果指定渲染文件，则直接加载
    if file.suffix.lower() == TextVideo.SUFFIX:
        video = TextVideo.from_text_video_file(file)
    else:  # 其余军假定为视频文件
        # 渲染文件将要存放的位置
        tv_file = file.parent / f'{file.name}{TextVideo.SUFFIX}'
        # 如果渲染文件存在
        if tv_file.exists():
            # 读取宽度信息
            w = TextVideo.read_info(tv_file)[TextVideo.ATTR_INDEX['w']]
            # 渲染文件宽度与指定宽度不符，则重新渲染
            if w != n:
                video = TextVideo.from_real_video_file(file, n, threshold, image_inverse)
                video.save(tv_file)
            else:
                # 直接加载渲染文件
                video = TextVideo.from_text_video_file(tv_file)
        else:
            video = TextVideo.from_real_video_file(file, n, threshold, image_inverse)
            video.save(tv_file)
    player = TextVideoPlayer()
    player.play(video, info)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='视频播放')
    parser.add_argument('file', type=str, help='视频地址')
    parser.add_argument('-n', type=int, help='每行字符数', default=50)
    parser.add_argument('--threshold', type=int, help="二值化阈值，0到255", default=127)
    parser.add_argument('--image_inverse', action="store_true", help="图像反相，黑白颠倒")
    parser.add_argument('--info', action="store_true", help="显示详细信息")

    args = parser.parse_args().__dict__
    play(**args)
