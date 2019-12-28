from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import random
from collections import deque
from time import time

def draw_ch_zn(img, str, font, loc, color):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text((loc[0], loc[1]), str, color, font)
    return np.array(pil_img)


class BulletScreen:
    def __init__(self, text_set, fontPath, resolution, time=3, fps=30, enlarge=10):
        self.font = ImageFont.truetype(fontPath, 20, encoding='utf-8')
        self.texts = list(text_set) * enlarge
        self.resolution = resolution
        self.posadd = self.resolution[0] / (time * fps)
        color = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
        self.conf = []
        for i in range(len(self.texts)):
            conf = []
            index = random.randint(0, 3)
            pos = random.randint(int(self.resolution[1]*0.1),
                                 int(self.resolution[1]*0.9+1))
            start = self.resolution[0]
            conf.append(i)
            conf.append(color[index])
            conf.append(pos)
            conf.append(start)
            self.conf.append(conf)
        random.shuffle(self.conf) # 就按照此顺序
        self.noSent = deque(self.conf)
        self.sentOut = deque([])
        self.isSent = True
        self.flag = 0

    def __call__(self, frame):
        if self.isSent and len(self.noSent) > 0:
            # 编号，颜色，高度位置（不变), x轴位置
            sample = self.noSent.popleft()
            self.sentOut.append(sample)
            self.isSent = False
        self.flag += 1
        self.flag %= 2
        if self.flag == 0:
            self.isSent = True
        for i in range(len(self.sentOut)):
            sample = self.sentOut[i]
            text = self.texts[sample[0]]
            frame = draw_ch_zn(frame, text, self.font, (int(sample[3]), sample[2]), sample[1])
            self.sentOut[i][3] = sample[3] - self.posadd
        i = 0
        while len(self.sentOut):
            sample = self.sentOut[i]
            if sample[3] < - 0.5 * self.resolution[0]:
                self.sentOut.popleft()
            else:
                break
        return frame


if __name__=='__main__':
    text = ['你好', '我要给你生猴子', '帅气逼人']
    fontPath = '../config/simhei.ttf'
    frame = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
    bullet = BulletScreen(text, fontPath, frame.shape[:2][::-1])
    while 1:
        a = time()
        result = bullet(frame.copy())
        cv2.imshow('res', result)
        print((time() - a))
        cv2.waitKey(10)
        if len(bullet.sentOut) == 0:
            break










