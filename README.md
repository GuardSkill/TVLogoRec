## 数据生成PipeLine

修改参数、路径，并运行以下脚本

1.裁剪RGBA的台标图像：

```
python ./TVDataGenerator/crop_RGBA_ROI.py
```



2.如果类别增加或减少,生成相应的yaml配置文件：

```
python ./TVDataGenerator/class_id.py
```



3.数据生成：

```
python ./TVDataGenerator/DataGenV3.py
```





4.修改[TVlogoDynamic.yaml](yolov5-master/data/TVlogoDynamic.yaml) 文件夹中数据集的路径，并训练：

```
python ./yolov5-master/train.py --img 640 --batch-size 14 --data TVlogoDynamic.yaml --weight yolov5m.pt --epoch 20
```

