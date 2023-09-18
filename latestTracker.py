import os
from IPython import display
import ultralytics
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2


HOME = os.getcwd()
print(HOME)

SOURCE_VIDEO_PATH = f"{HOME}/vehicle-counting.mp4"
MODEL = "yolov8x.pt"
# settings
LINE_START = sv.Point(50, 1500)
LINE_END = sv.Point(3840-50, 1500)

TARGET_VIDEO_PATH = f"{HOME}/vehicle-counting-result-with-counter.mp4"


display.clear_output()
ultralytics.checks()
print("supervision.__version__:", sv.__version__)

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - car, motorcycle, bus and truck
#selected_classes = [2, 3, 5, 7]
selected_classes = [2]
line_points = []

def draw_line(event, x, y, flags, param):
    global line_points

    if event == cv2.EVENT_LBUTTONDOWN:
        line_points.append((x, y))

        if len(line_points) == 2:
            cv2.line(frame, line_points[0], line_points[1], (0, 255, 0), 2)
            cv2.imshow('Frame', frame)
            # Now line_points[0] and line_points[1] contain the coordinates of the line
            # You can use these coordinates elsewhere in your script as needed
            # Reset the line_points list to allow drawing a new line
            line_points = []

def showOneFrame():
    global frame, line_points
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
    frame = anotated_frame
    #sv.plot_image(anotated_frame, (16,16))
    cv2.imshow('Frame', anotated_frame)
    cv2.setMouseCallback('Frame', draw_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh= 0.25, track_buffer = 30,match_thresh = 0.8,frame_rate =30)

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]

    box_annotated_frame=box_annotator.annotate(scene=frame.copy(),
                                    detections=detections,
                                    labels=labels)
    # update line counter
    line_zone.trigger(detections)
    print(line_zone.in_count, line_zone.out_count)
    # return frame with box and line annotated result
    return  line_zone_annotator.annotate(box_annotated_frame, line_counter=line_zone)

showOneFrame()

# process the whole video
# sv.process_video(
#     source_path = SOURCE_VIDEO_PATH,
#     target_path = TARGET_VIDEO_PATH,
#     callback=callback
# )