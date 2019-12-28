import torch
import numpy as np
import cv2
from config.flags import save_background_to_flags
from utils.textSlide import BulletScreen
def blur_background(frame, kernel_size_f, fps):
    if kernel_size_f < 20: # 没有模糊完毕
        kernel_size_f = kernel_size_f + 20/fps
        kernel_size = int(kernel_size_f)
        background = frame.copy()
        if kernel_size != 0:
            background = cv2.blur(frame, (kernel_size, kernel_size))
    else:
        kernel_size_f = 20
        background = cv2.blur(frame, (kernel_size_f, kernel_size_f))
    return background, kernel_size_f

def clear_background(frame, kernel_size_f, fps=5):
    if kernel_size_f > 0:  # 没有清晰完毕
        kernel_size_f = kernel_size_f - 20/fps
        kernel_size = int(kernel_size_f)
        background = frame.copy()
        if kernel_size != 0:
            background = cv2.blur(frame, (int(kernel_size), int(kernel_size)))
    else:
        kernel_size_f = 0
        background = frame.copy()
    return background, kernel_size_f

def is_last_for_gesture(gesture, recoders):
    leng = len(recoders)
    count = 0
    limit = 3
    if gesture == '5':
        limit = leng - 1
    for i in range(-1, -leng-1, -1):
        if recoders[i] == gesture:
            count += 1
    assert count >= 1
    if gesture == '5':
        if count > limit:
            return True
        if count == 1:
            return False
        else:
            return None

    if count == limit:
        return False
    elif count > limit:
        return True
    else:
        return None

def gestures_map_to_behaviors(frame, gesture, models_list, flags, points):
    # 先获得人物分割mask
    segnet = models_list[0]
    mask = segnet.predict(frame.copy())
    y, x = np.where(mask > 0.5)
    flags.recoders.append(gesture)
    flags.oriFrame = frame.copy()
    flags.pos = [x, y]
    last = is_last_for_gesture(gesture, flags.recoders)
    if last is None:  # 被抛弃，还没到触发线
        gesture_two(flags, frame)
        gesture_three(flags, frame)
        synthesized_img = gesture_one(frame, flags, (y, x))
        synthesized_img = gesture_four(synthesized_img, flags, models_list[1])
        return synthesized_img

    if gesture == '1':  # blur or back to clear. 渐进效果
        if flags.blur.kernel_size == 0 and not last:
            flags.blur.flag = True
        elif flags.blur.kernel_size == 20 and not last:
            flags.blur.flag = False

    if gesture == '2': # 背景变化，同时保持虚化程度
        if not last: # 背景变换
            flags.background.flag = True  # 背景不是原始背景了
            if flags.videos.type == 'video':
                if flags.videos.index is None:
                    flags.videos.index = 0
                else:
                    flags.videos.index = (flags.videos.index + 1) % flags.videos.numOfVideos
                flags.videos.captures[flags.videos.index].set(cv2.CAP_PROP_POS_FRAMES, 1)
            else:
                if flags.videos.iindex is None:
                    flags.videos.iindex = 0
                else:
                    flags.videos.iindex = (flags.videos.iindex + 1) % flags.videos.numOfImgs

    if gesture == '3':
        if not last: # 黑白效果
            flags.gray.flag = not flags.gray.flag
    if gesture == '4':
        if not last:
            flags.decoration.flag = not flags.decoration.flag

    if gesture == '5':
        if not last:  # 有过中断
            flags.extend.locked = True
            print('allow locked')
        if last:
            # 记录手心位置
            if flags.extend.locked:
                print('new pos')
                flags.extend.pos = points.mean(axis=0)
                flags.extend.locked = False
            elif flags.extend.pos is not None:
                current_pos = points.mean(axis=0)
                x_offset = abs(current_pos[0] - flags.extend.pos[0])  # <0 ? >0
                x_offset /= flags.extend.scale
                if x_offset > 1:
                    x_offset = 1
                flags.extend.offset = int(x_offset * flags.resolution[0])
    if gesture == '8':
        if not last:
            if flags.bullet.flag is None:
                flags.bullet.flag = BulletScreen(flags.bullet.text, flags.bullet.fontPath,
                                                 (flags.resolution[0]*3,flags.resolution[1],),
                                                time=5, fps=flags.fps)
            else:
                pass
                # if len(flags.bullet.flag.sentOut) == 0:
                #     flags.bullet.flag.sentOut = None
    if gesture == 'OK':
        if not last:
            flags.endding = True

    if gesture == 'GOOD':
        if not last:
            if flags.videos.type == 'img':
                flags.videos.type = 'video'
            else:
                flags.videos.type = 'img'
        print(flags.videos.type)
    gesture_two(flags, frame)
    gesture_three(flags, frame)
    synthesized_img = gesture_one(frame, flags, (y, x))
    synthesized_img = gesture_four(synthesized_img, flags, models_list[1])
    return synthesized_img



def gesture_two(flags, frame):
    if flags.videos.type == 'video':
        if flags.videos.index is None:
            return
    else:
        if flags.videos.iindex is None:
            return
    save_background_to_flags(flags, frame)

def gesture_three(flags, frame):
    if flags.gray.flag: # convert to gray
        background = flags.background.img
        if background is None:
            background = frame
        if len(background.shape) == 3:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        background = cv2.equalizeHist(background)
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        flags.background.img = background
    else:
        pass


def gesture_one(frame, flags, pos):
    if flags.background.img is not None:
        background = flags.background.img
    else:
        background = frame
    if flags.blur.flag:  # blur
        background, kernel_size = blur_background(background, flags.blur.kernel_size, flags.fps)
        flags.blur.kernel_size = kernel_size
        synthesized_img = background.copy()
        synthesized_img[pos[0], pos[1]] = frame[pos[0], pos[1]]
    else:  # clear
        background, kernel_size = clear_background(background, flags.blur.kernel_size, flags.fps)
        flags.blur.kernel_size = kernel_size
        synthesized_img = background.copy()
        synthesized_img[pos[0], pos[1]] = frame[pos[0], pos[1]]

    return synthesized_img

def gesture_four(frame, flags, faceDet):
    if flags.decoration.flag:
        _, lm = faceDet(frame, threshold=0.5) # dets, lms, dets不需要
        # lm[4] lm[5] 贴猫鼻
        # (1,4), (2,5) 分别计算左右位置猫脸三根毛
        # cv2.circle(frame, (int(lm[4]), int(lm[5])), 2, (0, 0, 255), -1)
        nosePos = [lm[4], lm[5]]
        leftMouthPos = [lm[6], lm[7]]
        distance = cal_euler_dist(nosePos, leftMouthPos)
        scale = distance / flags.decoration.distLeft * 1.5
        catImg = flags.decoration.catFace.copy()
        catImg = cv2.resize(catImg, dsize=None, dst=None, fx=scale, fy=scale)
        # catNosePos = flags.decoration.nosePos * scale
        catMaskPos_x = (flags.decoration.maskPos[0] * scale).astype(np.int64)
        catMaskPos_y = (flags.decoration.maskPos[1] * scale).astype(np.int64)
        maskPos_x = (flags.decoration.relativePos[0] * scale + nosePos[0]).astype(np.int64)
        maskPos_y = (flags.decoration.relativePos[1] * scale + nosePos[1]).astype(np.int64)
        maskPos_x = np.clip(maskPos_x, 0, flags.resolution[0]-1)
        maskPos_y = np.clip(maskPos_y, 0, flags.resolution[1]-1)
        frame[maskPos_y, maskPos_x] = catImg[catMaskPos_y, catMaskPos_x]
        # cv2.circle(catImg, (int(catNosePos[0]), int(catNosePos[1])), 2, (255, 0, 0), -1)
        # cv2.imshow('mask', catImg)
        return frame
    else:
        return frame

def cal_euler_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.square(p1 - p2).sum())