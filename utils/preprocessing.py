import cv2
import numpy as np
import torch
import torch.nn as nn

def Normalize_Img(imgOri, scale, mean, val):
    img = np.array(imgOri.copy(), np.float32)/scale
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = (img[j,:,:,i]-mean[i])*val[i]
        return img
    else:
        for i in range(len(mean)):
            img[:,:,i] = (img[:,:,i]-mean[i])*val[i]
        return img

def padding_img(img_ori, size=224, color=128):
    height = img_ori.shape[0]
    width = img_ori.shape[1]
    img = np.zeros((max(height, width), max(height, width), 3)) + color

    if (height > width):
        padding = int((height - width) / 2)
        img[:, padding:padding + width, :] = img_ori
    else:
        padding = int((width - height) / 2)
        img[padding:padding + height, :, :] = img_ori

    img = np.uint8(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.array(img, dtype=np.float32)

def resize_padding(image, dstshape, padValue=0):
    height, width, _ = image.shape
    ratio = float(width) / height  # ratio = (width:height)
    # 长边为224，保持长宽比得到短边长度
    dst_width = int(min(dstshape[1] * ratio, dstshape[0]))
    dst_height = int(min(dstshape[0] / ratio, dstshape[1]))
    origin = [int((dstshape[1] - dst_height) / 2), int((dstshape[0] - dst_width) / 2)]
    if len(image.shape) == 3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        newimage = np.zeros(shape=(dstshape[1], dstshape[0], image.shape[2]), dtype=np.uint8) + padValue
        newimage[origin[0]:origin[0] + dst_height, origin[1]:origin[1] + dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1] + dst_width, origin[0] + dst_height]  # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height), interpolation=cv2.INTER_NEAREST)
        newimage = np.zeros(shape=(dstshape[1], dstshape[0]), dtype=np.uint8)
        newimage[origin[0]:origin[0] + height, origin[1]:origin[1] + width] = image_resize
        bbx = [origin[1], origin[0], origin[1] + dst_width, origin[0] + dst_height]  # x1,y1,x2,y2
    return newimage, bbx

def generate_input(exp_args, inputs, prior=None):
    inputs_norm = Normalize_Img(inputs, scale=exp_args.img_scale, mean=exp_args.img_mean, val=exp_args.img_val)

    if exp_args.video == True:
        if prior is None:
            prior = np.zeros((exp_args.input_height, exp_args.input_width, 1))
            inputs_norm = np.c_[inputs_norm, prior]
        else:
            prior = prior.reshape(exp_args.input_height, exp_args.input_width, 1)
            inputs_norm = np.c_[inputs_norm, prior]

    inputs = np.transpose(inputs_norm, (2, 0, 1))
    return np.array(inputs, dtype=np.float32)



def crop_init(frame):
    height, width = frame.shape[:2]
    center_h, center_w = height // 2, width // 2
    start_h, end_h = center_h - 360 // 2, center_h + 360 // 2
    start_w, end_w = center_w - 240 // 2, center_w + 240 // 2
    crop_frame = frame[start_h:end_h, start_w:end_w, :]
    return crop_frame.copy()