# -*- coding: utf-8 -*-
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import tensorflow as tf
import numpy as np
from efficientnet import tfkeras as efn 
from array import array
import paho.mqtt.client as mqtt
import configparser
import datetime
import threading
import base64,io
import ast, re
import PIL.Image as Image
import paho.mqtt.publish as publish


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.chdir('/home/pi/yolov5')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def makeDict(Message):
    pattern = "({.*})"
    Message_String = str(Message)
    Message_Dict = ast.literal_eval(re.findall(pattern,Message_String)[0])
    return Message_Dict

def tobase64(img):
    return base64.b64encode(img).decode('ascii')

def readimage(path):
    count = os.stat(path).st_size / 2
    with open(path, "rb") as f:
        return f.read()

def imgArraytobase64(imgArray):
    pil_im = Image.fromarray(imgArray)
    b = io.BytesIO()
    pil_im.save(b, 'jpeg')
    im_bytes = b.getvalue()
    b64Img = tobase64(im_bytes)
    return b64Img

def on_connect(client, userdata, flags, rc):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+" Connected with result code "+str(rc))
    client.subscribe(ToMQTTTopicSUB)
    
def on_disconnect(client, userdata, rc):
    client.reconnect()
    client.subscribe(ToMQTTTopicSUB)

def on_message(client, userdata, msg):
    print("message Received")

    data_dict = makeDict(msg.payload)

    print("Image Received")
    try:
        dec = data_dict['device']
        rep = data_dict['reply']
        nparr = np.frombuffer(base64.b64decode(data_dict['img']), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        global abcmodel,efn_model,badmodel
        threading.Thread(target=dect_pic,args = (abcmodel,efn_model,badmodel,img_np,dec,rep,640)).start()
    except Exception as e:
#         pub_topic = {"pi":config['Mqtt']['pi'],"line":config['Mqtt']['line'],"arm":config['Mqtt']['arm']}
        publish.single("fromai/", str({"result":"ERROR!","errorcode":e}), hostname=ToMQTTTopicServerIP,port=ToMQTTTopicServerPort) #傳輸MQTT訊息 單次傳輸方便用




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def load_Yolo_model(weights = './last.pt'):
    device = select_device('cpu')
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(640, s=model.stride.max())
    half = device.type != 'cpu'
    if half:
        model.half()
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    return model

def load_CNN_model(weights = './efnB7.hdf5'):
    model = tf.keras.models.load_model(weights)
    return model

def dect_pic(model,efn_model,badmodel,im0s,dec,reply,imgsz=640):
    write_bbox_to_img = True

    device = select_device('cpu')
    half = device.type != 'cpu'
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # out = './output'
#     im0s = cv2.imread(img_path)
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment='')[0]

    # Apply NMS .4:conf  .5:iou
    pred = non_max_suppression(pred, 0.4, 0.5, classes='', agnostic='')
    # print(pred)

    # Process detections
    detect_list = []
    result_list = []
    for i, det in enumerate(pred):  # detections per image
        s, im0 =  '', im0s
        
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                de = []
                for i in xyxy:
                    print(i/10)
                    de.append(int(i/10))

                if(detect_list == []):
                    crop_img = im0[c1[1]:c2[1], c1[0]:c2[0]] #crop_img's ndarray
                    im = Image.fromarray(crop_img).resize((600,600)) #read from array and resize resize((size,size)) size:efn_model's 
                    test = np.array(im).reshape(1,600,600,3)/255
                    pre = efn_model.predict(test)

                    pred=np.argmax(pre,axis=1)[0]
                    use_badmango_yolo = True
                    if use_badmango_yolo:
    #                     print(crop_img.shape)
                        im = Image.fromarray(crop_img).resize((1280,720))
                        bad_im0 = dect_badpic(badmodel,np.array(im),640)
                        print(bad_im0.shape)
                        new_im = Image.fromarray(bad_im0).resize((crop_img.shape[1],crop_img.shape[0]))
                        im0[c1[1]:c2[1], c1[0]:c2[0]] = new_im
                    if write_bbox_to_img:  # Add bbox to image
                        result_list.append(names[int(pred)])
                        label = '%s %.2f' % (names[int(pred)], pre[0][int(pred)]) #e.g A 0.59
                        yoloLab = '%s %.2f' % (names[int(cls)], conf) #e.g A 0.50
                        print("CNN",label)
                        print("yolo",yoloLab)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        #             cv2.imwrite('./crop_dect.jpg', crop_img)
                else:
                    for i in detect_list:
                        if(i == de):
                            continue
                        else:
                            crop_img = im0[c1[1]:c2[1], c1[0]:c2[0]] #crop_img's ndarray
                            im = Image.fromarray(crop_img).resize((600,600)) #read from array and resize resize((size,size)) size:efn_model's 
                            test = np.array(im).reshape(1,600,600,3)/255
                            pre = efn_model.predict(test)

                            pred=np.argmax(pre,axis=1)[0]
                            use_badmango_yolo = True
                            if use_badmango_yolo:
            #                     print(crop_img.shape)
                                im = Image.fromarray(crop_img).resize((1280,720))
                                bad_im0 = dect_badpic(badmodel,np.array(im),640)
                                print(bad_im0.shape)
                                new_im = Image.fromarray(bad_im0).resize((crop_img.shape[1],crop_img.shape[0]))
                                im0[c1[1]:c2[1], c1[0]:c2[0]] = new_im
                            if write_bbox_to_img:  # Add bbox to image
                                result_list.append(names[int(pred)])
                                label = '%s %.2f' % (names[int(pred)], pre[0][int(pred)]) #e.g A 0.59
                                yoloLab = '%s %.2f' % (names[int(cls)], conf) #e.g A 0.50
                                print("CNN",label)
                                print("yolo",yoloLab)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        b64Img = imgArraytobase64(im0[:,:,::-1])  #convert imgarray to btye then to base64
        cv2.imwrite('./test_dect22.jpg', im0)
        data_dict = {"result":result_list,"img": b64Img,"reply":reply,"device":dec}
        config = configparser.ConfigParser()
        config.read("./config.ini")
        ToMQTTTopicServerIP = config['Mqtt']['MQTTTopicServerIP']
        ToMQTTTopicServerPort = int(config['Mqtt']['port'])
        pub_topic = {"pi":config['Mqtt']['pi'],"line":config['Mqtt']['line'],"arm":config['Mqtt']['arm']}
        publish.single(pub_topic[dec], str(data_dict), hostname=ToMQTTTopicServerIP,port=ToMQTTTopicServerPort) #傳輸MQTT訊息 單次傳輸方便用
        

def dect_badpic(model,im0s,imgsz=640):
    write_bbox_to_img = True
    device = select_device('cpu')
    half = device.type != 'cpu'
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # out = './output'
#     print(np.array(im0s)[:,:,::-1])
#     print(im0s.shape)
#     im0s = cv2.imread('./badmango.jpg')
#     print(im0s)
#     print(im0s.shape)
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment='')[0]

    # Apply NMS .4:conf  .5:iou
    pred = non_max_suppression(pred, 0.4, 0.5, classes='', agnostic='')
    # print(pred)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 =  '', im0s
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                if write_bbox_to_img:  # Add bbox to image
                    yoloLab = '%s %.2f' % (names[int(cls)], conf) #e.g A 0.50
                    print("yolo",yoloLab)
                    plot_one_box(xyxy, im0, label=yoloLab, color=colors[int(cls)], line_thickness=3)
        return im0


#讀取設定檔案
config = configparser.ConfigParser()
config.read("./config.ini")
_g_cst_ToMQTTTopicServerIP = config['Mqtt']['MQTTTopicServerIP']
_g_cst_ToMQTTTopicServerPort = int(config['Mqtt']['port'])
_yolo_mango_weight_path = config['Weights']['yolo_abc']
_yolo_badMango_weight_path = config['Weights']['yolo_bad']
_CNN_weight_path = config['Weights']['cnn']
ToMQTTTopicSUB = config['Mqtt']['sub']
#預讀模型
abcmodel = load_Yolo_model(_yolo_mango_weight_path) 
badmodel = load_Yolo_model(_yolo_badMango_weight_path)
efn_model = load_CNN_model(_CNN_weight_path)

#建立MQTT連線
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect
client.connect(_g_cst_ToMQTTTopicServerIP, _g_cst_ToMQTTTopicServerPort,60)
client.loop_forever()

