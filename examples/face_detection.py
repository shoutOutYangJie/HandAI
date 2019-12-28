import cv2
import sys
sys.path.append('../networks/')
from centerface import CenterFace
import numpy as np

def get_relative_pos(img, pos):
    y, x, _ = np.where(img != np.array([255, 255, 255]))
    maskPos = [x, y]
    y, x = y - pos[1], x - pos[0]
    return maskPos, [x, y]

def cal_euler_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.sqrt(np.square(p1 - p2).sum())

def get_decoration():
    decoration = cv2.resize(cv2.imread('../data/decoration/cat_decoration.png'), (512, 512))
    nosePos = np.array([1051, 1275]) * (512 / 2000)
    leftMouthPos = np.array([703, 1659]) * (512 / 2000)
    distLeft = cal_euler_dist(nosePos, leftMouthPos)
    maskPos, relativePos = get_relative_pos(decoration, nosePos,)
    return decoration, nosePos, leftMouthPos, distLeft, maskPos, relativePos


def test_image():
    frame = cv2.imread('../data/example_data/face.jpg')
    ori_frame = frame.copy()
    h, w = frame.shape[:2]
    resolution = (w, h)
    landmarks = True
    centerface = CenterFace(h, w, landmarks=landmarks, model_path='../ckpts/centerFace/centerface.onnx')
    dets, lms = centerface(frame, threshold=0.35)
    boxes, score = dets[:4], dets[4]
    cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        cv2.circle(frame, (int(lms[0]), int(lms[1])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lms[2]), int(lms[3])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lms[4]), int(lms[5])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lms[6]), int(lms[7])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(lms[8]), int(lms[9])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)

    # add decoration
    decoration, nosePos, leftMouthPos, distLeft, maskPos, relativePos = get_decoration()
    nosePos = [lms[4], lms[5]]
    leftMouthPos = [lms[6], lms[7]]
    distance = cal_euler_dist(nosePos, leftMouthPos)
    scale = distance / distLeft * 1.2
    catImg = decoration.copy()
    catImg = cv2.resize(catImg, dsize=None, dst=None, fx=scale, fy=scale)
    # catNosePos = flags.decoration.nosePos * scale
    catMaskPos_x = (maskPos[0] * scale).astype(np.int64)
    catMaskPos_y = (maskPos[1] * scale).astype(np.int64)
    maskPos_x = (relativePos[0] * scale + nosePos[0]).astype(np.int64)
    maskPos_y = (relativePos[1] * scale + nosePos[1]).astype(np.int64)
    maskPos_x = np.clip(maskPos_x, 0, resolution[0] - 1)
    maskPos_y = np.clip(maskPos_y, 0, resolution[1] - 1)
    ori_frame[maskPos_y, maskPos_x] = catImg[catMaskPos_y, catMaskPos_x]
    cv2.imshow('add decoraion', ori_frame)
    cv2.waitKey(0)


if __name__== '__main__':
    test_image()