import numpy as np
from config import handTrackConfig as htconf
from utils import hand_track_utils as htutils
import cv2

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

threshold = [
    (-0.90, -1), # 0, 1, 2, 3
    (0, -0.85, -1), # 0, 1, 2
    (0.7, -0.85, -1), # 0, 1, 2
    (0.7, -0.85, -1), # 0, 1, 2
    (0.7, -0.85, -1) # 0, 1, 2
]

pos = [
    [(3, 2, 1), (4, 3, 2)],
    [(8, 6, 0)],
    [(12, 10, 0)],
    [(16, 14, 0)],
    [(20, 18, 0)]
]

record = {
    # '0': [[0, 0, 0, 0, 0]],
    '1': [[0, 2, 0, 0, 0]],
    '2': [[0, 2, 2, 0, 0]],
    '3': [[0, 2, 2, 2, 0]],
    '4': [[0, 2, 2, 2, 2]],
    '5': [[1, 2, 2, 2, 2]],
    'OK': [[0, 0, 2, 2, 2]],
    'GOOD': [[1, 0, 0, 0, 0]],
    '8': [[1, 2, 0, 0, 0], [1, 1, 0, 0, 0]],
}

def cal_finger_angle(points):
    res = []
    for p in pos:
        temp = []
        for i in p:
            start, mid, end = i
            v1 = points[start] - points[mid]
            v1 /= np.linalg.norm(v1)
            v2 = points[end] - points[mid]
            v2 /= np.linalg.norm(v2)
            cos_ang = v1.dot(v2)
            temp.append(cos_ang)
        res.append(sum(temp) / len(temp))
    # print(res)
    return res

def recog_gesture(points):
    conf = cal_finger_angle(points)
    res = []
    for i, pred in enumerate(conf):
        thre = threshold[i]
        for c, t in enumerate(thre):
            if pred > t:
                res.append(c)
                break
    # print(res)
    for k, v in record.items():
        for v1 in v:
            if v1 == res:
                return k
    return None

def get_vis_gesture_map(map, points, vis_window_shape):
    points[:, 0] -= points[:, 0].min()
    points[:, 1] -= points[:, 1].min()
    points[:, 0] /= points[:, 0].max()
    points[:, 1] /= points[:, 1].max()
    points += 0.1
    points *= 0.8 * vis_window_shape[0]
    for i, point in enumerate(points):
        x, y = point
        cv2.circle(map, (int(x), int(y)), htconf.THICKNESS, htconf.POINT_COLOR, htconf.THICKNESS)
    for connection in htconf.connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        cv2.line(map, (int(x0), int(y0)), (int(x1), int(y1)), htconf.CONNECTION_COLOR, htconf.THICKNESS)
    return map
