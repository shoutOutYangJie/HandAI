from easydict import EasyDict
from collections import deque
import cv2
import os.path as osp
import os
import numpy as np

def init_flags():
    flags = EasyDict()
    flags.recoders = deque(maxlen=5)
    flags.fps = 5
    flags.count = 0
    flags.oriFrame = None
    flags.pos = None
    flags.resolution = (240, 360)

    flags.blur = EasyDict()
    flags.blur.flag = False
    flags.blur.kernel_size = 0

    flags.background = EasyDict()
    flags.background.img = None
    flags.background.flag = False

    flags.videos = EasyDict()
    flags.videos.lists = os.listdir('./data/videos')
    flags.videos.lists = [osp.join('./data/videos', i) for i in flags.videos.lists]
    flags.videos.captures = []
    for i in range(len(flags.videos.lists)):
        cap = cv2.VideoCapture(flags.videos.lists[i])
        if cap.isOpened():
            flags.videos.captures.append(cap)
        else:
            raise Warning('%s is not found.' % flags.videos.lists[i])
    flags.videos.index = None
    flags.videos.numOfVideos = len(flags.videos.captures)
    flags.videos.imgLists = os.listdir('./data/pictures')
    flags.videos.imgLists = [osp.join('./data/pictures', i) for i in flags.videos.imgLists]
    flags.videos.imgs = []
    for i in range(len(flags.videos.imgLists)):
        img = cv2.imread(flags.videos.imgLists[i])
        img = cv2.resize(img, tuple(flags.resolution))
        flags.videos.imgs.append(img)
    flags.videos.iindex = None
    flags.videos.numOfImgs = len(flags.videos.imgs)
    flags.videos.type = 'img'  # option  "img" "video"


    flags.gray = EasyDict()
    flags.gray.flag = False

    flags.extend = EasyDict()
    flags.extend.pos = None
    flags.extend.flag = False
    flags.extend.locked = True
    flags.extend.offset = 0
    flags.extend.scale = 0.15 * flags.resolution[0]

    flags.decoration = EasyDict()
    flags.decoration.flag = False
    flags.decoration.catFace = cv2.resize(cv2.imread('./data/decoration/cat_decoration.png'), (512, 512))
    flags.decoration.nosePos = np.array([1051, 1275]) * (512 / 2000)
    flags.decoration.leftMouthPos = np.array([703, 1659]) * (512 / 2000)
    flags.decoration.distLeft = cal_euler_dist(flags.decoration.nosePos, flags.decoration.leftMouthPos)

    flags.decoration.relativePos = get_relative_pos(flags.decoration.catFace, flags.decoration.nosePos,
                                                    flags)

    flags.bullet = EasyDict()
    flags.bullet.fontPath = './config/simhei.ttf'
    flags.bullet.text = ['小哥哥好帅气', 'python大法好', '我不要996', '在下国服第一锐雯', '既是程序猿又是单身狗',
                         'GitHub求个Star', '可怜可怜我给个Offer吧']
    flags.bullet.flag = None

    flags.endding = False
    return flags

def save_background_to_flags(flags, background):
    if flags.background.flag is not False:
        if flags.videos.type == 'video':
            cap = flags.videos.captures[flags.videos.index]
            ret, background = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            background = np.rot90(background, k=-1)
            flags.background.img = cv2.resize(background, tuple(flags.resolution))
        else:
            img = flags.videos.imgs[flags.videos.iindex]
            flags.background.img = img

    else:
        flags.background.img = background.copy()

def cal_euler_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.square(p1 - p2).sum())

def get_relative_pos(img, pos, flags):
    y, x, _ = np.where(img != np.array([255, 255, 255]))
    flags.decoration.maskPos = [x, y]
    y, x = y - pos[1], x - pos[0]
    return [x, y]
