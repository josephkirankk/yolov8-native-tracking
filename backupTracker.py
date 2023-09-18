import supervision as sv
from ultralytics import YOLO
import numpy as np
import time
import cv2


#REFERENCE : https://colab.research.google.com/drive/1pP8m0I9NfPBQt0FwEstLoRtRDbCYZYVd

cap = cv2.VideoCapture("abc.mp4")
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_fps = int(cap.get(cv2.CAP_PROP_FPS))

byte_tracker = BYTETracker(BYTETrackerArgs())

LINE_START = Point(180, 0)
LINE_END = Point(200, cap_height)

LINE_START_2 = Point(cap_width - 180, 0)
LINE_END_2 = Point(cap_width - 200, cap_height)

line_counter = LineZone(start=LINE_START, end=LINE_END)
line_counter2 = LineZone(start=LINE_START_2, end=LINE_END_2)

box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5,
)
line_annotator = LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

lst = []
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)
    for result in results:
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.confidence > 0.5]

        if len(detections) == 0:
            continue
        
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_ids = match_detections_with_tracks(
            detections=detections,
            tracks=tracks
        )
        detections.tracker_id = tracker_ids

        labels = [
            f"#{tracker_id} {model.model.names[class_id]}: {confidence:.2f}"
            for _, _, confidence, class_id, tracker_id in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # error area starts
        line_counter.trigger(detections)
        line_counter2.trigger(detections)

        frame = line_annotator.annotate(frame, line_counter)
        frame = line_annotator.annotate(frame, line_counter2)
        # error area ends