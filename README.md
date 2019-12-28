# HandAI
Using hand gestures control different effect of photography.

[Chinese Document](https://blog.csdn.net/qq_34914551/article/details/103746527)

# Introduction
In this repo, HandAI leverages eight gestures to take charge of eight functions, 
including bluring background, changing background, converting background to gray,
face decoration, extending frame, unbolcking bullet screen, video background mode, ending recording.
The key of HandAI is hand keypoint detection and portrait segmentation.

# Main dependencies
opencv-python >= 4.0  
TensorFlow2.0(GPU is unnecessary)  
PyTorch-GPU >= 1.1  
Numpy  
Pillow  

# Exhibitions
![one](https://github.com/shoutOutYangJie/HandAI/blob/master/pictures/one.gif)
![two](https://github.com/shoutOutYangJie/HandAI/blob/master/pictures/two.gif)
![four](https://github.com/shoutOutYangJie/HandAI/blob/master/pictures/four.gif)

The rest results of other gestures can be found at "./pictures"

# Using HandAI on Web Camera
Make sure that your computer has a camera, which "cv2.VideoCapture" needs.
```
cd HandAI-master
python webCamera_demo.py
```

# Using HandAI on a Video
Some settings need to change since fps of video is about 30, but HandAI only has fps of 5. 
The concrete instruction has been written at "save_video_demo.py".
```
cd HandAI-master
python save_video_demo.py
```

# Insturction about divided part
In the "./examples" file, you can find how to use divided part module of HandAI, which make you more clear about how HandAI works.

# Acknowledgement
[MediaPipe](https://github.com/google/mediapipe)  
[PortraitNet](https://github.com/dong-x16/PortraitNet)  
[CenterFace](https://github.com/Star-Clouds/CenterFace)
