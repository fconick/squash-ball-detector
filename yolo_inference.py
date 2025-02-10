from ultralytics import YOLO

#Choose a model 
# model = YOLO('yolov8x') #YOLO out of the box
model = YOLO('models/yolo5_v4_last.pt') # YOLO model trained with V4 of the dataset in roboflow

result = model.predict('input_videos/squash_image.png', conf = 0.01, save = True)
result = model.track('input_videos/squash_video2.mp4',   conf = 0.1, save = True)
# print(result)
