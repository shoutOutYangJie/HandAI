import cv2
import numpy as np
import torch
import torch.nn as nn
from utils.textSlide import draw_ch_zn
from PIL import Image, ImageFont, ImageDraw
import math

font = ImageFont.truetype('./config/simhei.ttf', 40, encoding='utf-8')
def postprocessing(img, vis_gesture, vis_window, flags):
    resolution = flags.resolution
    extend_shape = (resolution[1], resolution[0] * 3, 3)
    extend_img = np.zeros(shape=extend_shape, dtype=np.uint8)

    if flags.extend.pos is not None:
        cv2.circle(img, (int(flags.extend.pos[0]), int(flags.extend.pos[1])), 4, (0, 255, 0), 2)

    if flags.extend.flag is False:
        offset = flags.extend.offset
        extend_img[:, resolution[0]-offset:resolution[0]] = img[:, 0:offset]
        extend_img[:, 2*resolution[0]:2*resolution[0]+offset] = img[:, resolution[0]-offset:resolution[0]]
        if offset == flags.resolution[0]:
            flags.extend.pos = None
            flags.extend.flag = True
            flags.extend.offset = 0
    else:
        offset = flags.resolution[0] - flags.extend.offset
        extend_img[:, resolution[0] - offset:resolution[0]] = img[:, 0:offset]
        extend_img[:, 2 * resolution[0]:2 * resolution[0] + offset] = img[:, resolution[0] - offset:resolution[0]]
        if offset == 0:
            flags.extend.pos = None
            flags.extend.flag = False
            flags.extend.offset = 0

    mask = np.zeros_like(extend_img, dtype=np.uint8)
    if flags.bullet.flag is not None:
        mask = flags.bullet.flag(mask)
        x, y = flags.pos
        mask[y, x + resolution[0]] = [0, 0, 0]
        if flags.extend.flag:
            mask[y, x] = [0, 0, 0]
            mask[y, x + 2 * resolution[0]] = [0, 0, 0]
        if len(flags.bullet.flag.sentOut) == 0:
            flags.bullet.flag = None

    y, x, _ = np.where(mask != np.array([0, 0, 0]))
    extend_img[:, resolution[0]:2 * resolution[0]] = img
    extend_img[y, x] = mask[y, x]
    extend_img[0:vis_window[0], resolution[0]:resolution[0]+vis_window[1], :] = vis_gesture

    if flags.endding:
        flags.count += 1
        if flags.count >= flags.fps * 3:
            flags.count = 0
            return None
        second = int(math.ceil((flags.fps * 3 - flags.count) / flags.fps))
        str_show = '%d秒之后自动结束录制' % second
        w, h = font.getsize(str_show)
        w = (flags.resolution[0] * 3 - w) // 2
        h = (flags.resolution[1] - h) // 2
        extend_img = draw_ch_zn(extend_img, str_show, font, (w, h), color=(128, 128, 255))
    if flags.videos.type == 'video':
        w, h = flags.resolution
        w = int(w * 0.95)
        h = int(h * 0.05)
        cv2.circle(extend_img, (flags.resolution[0] + w, h), 5, (0,0, 255), -1)
        if flags.extend.flag:
            cv2.circle(extend_img, (w, h), 5, (0,0, 255), -1)
            cv2.circle(extend_img, (w + 2 * flags.resolution[0], h), 5, (0,0, 255), -1)
    return extend_img
