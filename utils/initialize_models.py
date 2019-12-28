from config import handTrackConfig as htconf
from utils import hand_track_utils as htutils
from yaml import load
import yaml
from networks.hand_tracker import HandTracker
from networks.centerface import CenterFace
from networks.model_mobilenetv2_seg_small import MobileNetV2
import torch

def models_init(flags):
    HandDet = HandTracker(
    htconf.PALM_MODEL_PATH,
    htconf.LANDMARK_MODEL_PATH,
    htconf.ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3)

    config_path = './config/portraitNet.yaml'
    with open(config_path, 'rb') as f:
        cont = f.read()
    cf = load(cont, Loader=yaml.FullLoader)
    portraitNet = MobileNetV2(n_class=2,
                              config=cf)
    bestModelFile = './ckpts/portraitNet/portraitNet.pth'
    checkpoint_video = torch.load(bestModelFile)
    portraitNet.load_state_dict(checkpoint_video, strict=False)
    portraitNet.cuda()

    centerface = CenterFace(flags.resolution[1], flags.resolution[0])

    return HandDet, portraitNet, centerface