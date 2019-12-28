import cv2
import numpy as np
from utils import hand_track_utils as htutils
from utils.initialize_models import models_init
from utils.utils import gestures_map_to_behaviors
from config.flags import init_flags
from utils.postprocessing import postprocessing

# Notice that ensure your video's direction, which means your video is vertical screen shot.
# So, the ratio of width and height should be nearly 1080/1920.
# In my setting, the ratio is 240/360.
# If your video is not vertical, please use np.rot90 to adjust it, which refer to Line 28.
# One more thing, if you want to test on video instead of web camera, to obtain more stable result,
# please set "flags.fps" as 30 at "./config/flags.py", set "maxlen" of "flags.recoders" as 10.
# please also set "limit" as 7 at "./utils/utils.py", Line 33.


def main():
    # adjust your video path.
    capture = cv2.VideoCapture('./data/xxxx.mp4')
    ret, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS)
    if capture.isOpened():
        hasFrame, frame = capture.read()
    else:
        hasFrame = False
    flags = init_flags()
    videoWriter = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (flags.resolution[0]*3, flags.resolution[1]))
    detector, segNet, faceDet = models_init(flags)
    while hasFrame:
        # frame = crop_init(frame)  # (360, 240, 3), 只针对电脑摄像头
        frame = np.rot90(frame, k=-1)
        frame = cv2.resize(frame, (240, 360))
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
        synthesized_img = postprocessing(synthesized_img, vis_gesture, vis_window_shape, flags)
        cv2.imshow("handTrack", synthesized_img)
        cv2.waitKey(10)
        hasFrame, frame = capture.read()
        videoWriter.write(synthesized_img)
        if synthesized_img is None:
            videoWriter.release()
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            videoWriter.release()
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()