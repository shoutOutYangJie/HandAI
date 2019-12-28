from networks.hand_tracker import HandTracker
from networks.model_mobilenetv2_seg_small import MobileNetV2
import cv2
import torch
import numpy as np
from config import handTrackConfig as htconf
from utils import hand_track_utils as htutils
from utils.initialize_models import models_init
from utils.utils import gestures_map_to_behaviors
from config.flags import init_flags, save_background_to_flags
from utils.preprocessing import crop_init
from utils.postprocessing import postprocessing



def main():
    capture = cv2.VideoCapture(0)

    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False
    flags = init_flags()

    detector, segNet, faceDet = models_init(flags)
    while hasFrame:
        frame = crop_init(frame)  # (360, 240, 3)

        points, _ = detector(frame.copy())
        gesture = None
        vis_window_shape = int(min(frame.shape[0], frame.shape[1]) * 0.3)
        vis_window_shape = (vis_window_shape, vis_window_shape, 3)
        vis_gesture = np.zeros(shape=vis_window_shape, dtype=np.uint8)
        if points is not None:
            gesture = htutils.recog_gesture(points)
            vis_gesture = htutils.get_vis_gesture_map(vis_gesture, points.copy(), vis_window_shape)

        synthesized_img = gestures_map_to_behaviors(frame, gesture, [segNet, faceDet],
                                                        flags, points)
        synthesized_img = postprocessing(synthesized_img,vis_gesture, vis_window_shape, flags)
        if synthesized_img is None:
            break
        cv2.imshow("handTrack", synthesized_img)
        hasFrame, frame = capture.read()
        key = cv2.waitKey(20)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()