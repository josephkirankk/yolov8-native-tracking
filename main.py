import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np



LINE_START = sv.Point(320, 0)
LINE_END = sv.Point(320, 480)


def main():
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )

    model = YOLO("yolov8n.pt")
    rtsp_stream_url = "rtsp://joseph:joseph@192.168.1.120:554/stream1"
    
    #source can by equal to =0, =1, =2, etc. for webcam
    for result in model.track(source=rtsp_stream_url, show=True, stream=True, agnostic_nms=True):
        
        #frame = result.orig_img
        #detections = sv.Detections.from_yolov8(result)
        
        #result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        detections = detections[detections.class_id == 0]
        detections = detections[detections.confidence > 0.5]
        detections = detections[detections.area > 1000]
        #detections = detections[(detections.class_id != 60) & (detections.class_id != 0)]

        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()