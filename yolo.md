# YOLOv8

## 安裝yolo

```shell
pip install ultralytics
```

## Datasets準備

1. roboflow下載datasets
2. 解壓縮放到自己想放的位置
3. 在workspace創立一個名叫datasets的資料夾
4. 將roboflow資料夾當中的train、valid、test資料夾移到datasets資料夾當中
5. 到roboflow資料夾當中的data.yaml當中修改train、val、test的資料路徑(使用絕對路徑)

## 訓練模型

``` shell
yolo task=detect mode=train model=<yolov8n/yolov8s/yolov8m>.pt imgsz=640 epochs=100 data=data.yaml的路徑
```
