#python train.py --img 640 --data TVlogo802.yaml --weights '' --cfg yolov5m.yaml
#python train.py --img 640 --data TVlogo802.yaml --weights 'yolov5m.pt'
#python train.py --img 640 --batch-size 32 --data TVlogo802.yaml --weights 'runs/train/V1_recall_0.31/weights/last.pt'  #22 server


python train.py --img 640 --batch-size 14 --data TVlogo57.yaml --weight yolov5m.pt --epoch 20


# Test command
#python detect_V3.py --source /Disk1/Video/东方卫视_1080_Bili.mp4 --weight ../V3_epoch_13.pt       # 海南新闻频道.mp4
# ./test/images      #./test_1080/images   #/Disk1/Video/海南卫视.mp4
# /Disk1/Dataset/TV_logo_data/Test/Renew_logo_Test
# python detect_V3.py --source /Disk1/Dataset/TV_logo_data/Test/logo_Test_1080P --weight ../V3_5_best.pt --conf-thres 0.5