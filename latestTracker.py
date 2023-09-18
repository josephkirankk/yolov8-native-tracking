import os
from IPython import display
import ultralytics
from ultralytics import YOLO
import supervision as sv
import numpy as np


HOME = os.getcwd()
print(HOME)

SOURCE_VIDEO_PATH = f"{HOME}/vehicle-counting.mp4"
MODEL = "yolov8x.pt"



display.clear_output()
ultralytics.checks()
print("supervision.__version__:", sv.__version__)

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - car, motorcycle, bus and truck
selected_classes = [2, 3, 5, 7]

def showOneFrame():
    # create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create instance of BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    # acquire first video frame
    iterator = iter(generator)
    frame = next(iterator)
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]

    # convert to Detections
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]

    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _ in detections
    ]

    # annotate and display frame
    anotated_frame=box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    sv.plot_image(anotated_frame, (16,16))



showOneFrame()