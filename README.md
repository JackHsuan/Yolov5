# Yolov5
tutorial of training yolov5 with some helpful tools in my experiment

1.clone yolov5

python>=3.8 Pytorch>=1.6 
```bash
git clone https://github.com/ultralytics/yolov5  # clone repo
curl -L -o tmp.zip https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip && unzip -q tmp.zip && rm tmp.zip  # download dataset
cd yolov5
pip install -qr requirements.txt  # install dependencies
```
2.make your own data

製作資料夾放影像、xml資料夾像下面這樣

xml製作:https://github.com/tzutalin/labelImg 
```bash
#訓練資料
dataset/train_img/{圖片} #001.jpg 002.jpg...XXX.jpg
dataset/train_anno/{xml檔案} #001.xml 002.xml...XXX.xml
dataset/train_txt/{圖片+txt} #yolo資料集後面會製作
#測試資料
dataset/test_img/{圖片}
dataset/test_anno/{xml檔案}
dataset/train_txt/{圖片+txt} #yolo資料集後面會製作
```
3.yaml

製作data.yaml放在dataset內
```bash
train: ./yolov5_config/train_data.txt #裡面存放dataset/train_txt/圖片路徑
val: ./yolov5_config/val_data.txt #裡面存放dataset/train_txt/圖片路徑
test: ./yolov5_config/test_data.txt #裡面存放dataset/test_txt/圖片路徑

# number of classes
nc: 3
# class names
names:['A','B','C']
```
3.利用image_aug.ipynb 製作Yolov5資料集

將image_aug.ipynb放在yolob5資料夾內

利用 1_labels_to_yolo_format 製作yolo資料集

利用 2_split_train_test:將圖片路徑寫入txt 製作yaml內的train.txt、val.txt、test.txt

製作anchors 將anchor產生的資料填入models/yolov5s.yaml

4.開始訓練
```bash
$ python train.py --data ./dataset/data.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
```
5.物件偵測 detect_withMQTT.py

預先載入Yolo模型、tensorflow CNN模型

連線至MQTTserver 訂閱頻道

頻道傳輸圖片(base64格式)後將圖片偵測後透過MQTT回傳偵測結果、圖片


