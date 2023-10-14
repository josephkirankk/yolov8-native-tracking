from ultralytics import YOLO
model = YOLO('pop_v2.1_best.pt') # load a pretrained YOLOv8n segmentation model
results = model.predict(source='image1.jpg') # predict on an image
print(results[0].boxes.data) # print img1 predictions (pixels