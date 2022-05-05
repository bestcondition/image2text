import argparse
import curses
import os
import time
from math import ceil
from pathlib import Path
from typing import List
from collections import deque
import multiprocessing

import cv2
from tqdm import tqdm

from text_image import TextImage, META_W

DEFAULT_ENCODING = 'utf-8'


def get_n(width):
    return ceil(width // META_W)


def image2text_for_pool(frame_arg):
    """
    多进程用

    :param frame_arg:
    :return:
    """
    frame, n, threshold, image_inverse = frame_arg
    return TextImage.from_array(frame, n, threshold, image_inverse)


class TextVideo:
    INDEX_ATTR = 'fps w h frame_number'.split()
    ATTR_INDEX = {
        attr: i
        for i, attr in enumerate(INDEX_ATTR)
    }

    def __init__(self, image_list: List[TextImage], fps):
        self.frames = image_list
        self.w = self.frames[0].w
        self.h = self.frames[0].h
        self.fps = fps
        self.T = 1 / fps

    @property
    def frame_number(self):
        return len(self.frames)

    def get_image(self, delta_t):
        try:
            return self.frames[int(delta_t / self.T)]
        except IndexError:
            return None

    def save(self, file):
        with open(file, mode='wt', encoding=DEFAULT_ENCODING) as fp:
            # 写入基础信息
            fp.write(' '.join(
                str(self.__getattribute__(attr))
                for attr in self.INDEX_ATTR
            ))
            # 写入每一帧
            for image in self.frames:
                fp.write('\n')
                fp.write(str(image))

    @classmethod
    def from_real_video_file(cls, real_video_file, width, threshold=127, image_inverse=False):
        # 每行字符数
        n = get_n(width)
        # 生成的图像队列
        text_image_list = []
        # 视频流读取
        video_capture = cv2.VideoCapture(str(real_video_file))
        # fps
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        # 总帧数，用来显示读取进度
        frame_number = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # tq = tqdm(total=frame_number)

        processes = max(1, os.cpu_count() - 1)

        frame_gen = (
            (video_capture.read()[1], n, threshold, image_inverse)
            for i in range(frame_number)
        )

        with multiprocessing.Pool(processes=processes) as pool:
            text_image_list = list(
                tqdm(pool.imap(image2text_for_pool, frame_gen), total=frame_number, desc='渲染中'))

        video_capture.release()
        return cls(
            image_list=text_image_list,
            fps=fps
        )

    @classmethod
    def from_text_video_file(cls, text_video_file):
        with open(text_video_file, mode='rt', encoding=DEFAULT_ENCODING) as fp:
            # 读取基础信息
            info_list = fp.readline().split()
            fps = float(info_list[cls.ATTR_INDEX['fps']])
            w = int(info_list[cls.ATTR_INDEX['w']])
            h = int(info_list[cls.ATTR_INDEX['h']])
            frame_number = int(info_list[cls.ATTR_INDEX['frame_number']])
            # 图像队列
            text_image_list = [
                TextImage([
                    fp.readline()[:w]
                    for j in range(h)
                ], copy=False)
                for i in range(frame_number)
            ]
            return cls(text_image_list, fps)


class TextVideoPlayer:
    def __init__(self, video: TextVideo):
        self.video = video
        self.count = 0

    def _get_fps_top(self, delta_t):
        try:
            return self.count / delta_t
        except ZeroDivisionError:
            return "inf"

    def play(self):
        self.count = 0
        curses.wrapper(self._play_process)

    def _play_process(self, std_scr):
        t0 = time.time()
        while True:
            std_scr.clear()
            self.count += 1
            delta_t = time.time() - t0
            image = self.video.get_image(delta_t)
            if image is None:
                break
            std_scr.addstr(str(image))
            std_scr.addstr(f"\nfps: {self._get_fps_top(delta_t)}")
            std_scr.refresh()
        std_scr.clear()
        std_scr.addstr("END")
        std_scr.refresh()
        std_scr.getch()


def play(file, width=200, threshold=127, image_inverse=False):
    file = Path(file)
    if file.suffix.lower() == '.txt':
        video = TextVideo.from_text_video_file(file)
    else:
        txt_file = file.parent / f'{file.stem}.temp.txt'
        if txt_file.exists():
            video = TextVideo.from_text_video_file(txt_file)
            if video.w != get_n(width):
                video = TextVideo.from_real_video_file(file, width, threshold, image_inverse)
                video.save(txt_file)
        else:
            video = TextVideo.from_real_video_file(file, width, threshold, image_inverse)
            video.save(txt_file)
    player = TextVideoPlayer(video)
    player.play()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='视频播放')
    parser.add_argument('file', type=str, help='视频地址')
    parser.add_argument('--width', type=int, help='视频宽，不是每行字符数哦，视频宽=每行字符数*2', default=20)
    parser.add_argument('--threshold', type=int, help="二值化阈值", default=127)
    parser.add_argument('--image_inverse', action="store_true", help="图像反相")

    args = parser.parse_args().__dict__
    play(**args)
