import os
from IPython import display
import ultralytics
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
import time
import threading


HOME = os.getcwd()
print(HOME)

SOURCE_VIDEO_PATH = f"{HOME}/videos/v1.mp4"
#SOURCE_VIDEO_PATH = "C:\JK\dev\repo\yolov8-native-tracking\videos\new\output_1696750021.mp4"

#MODEL = "astra_pop_nv1.pt"
#MODEL = "pop_v2.1_best.pt"
MODEL = "yolov8l.pt"
# settings
# LINE_START = sv.Point(50, 1500)
# LINE_END = sv.Point(3840-50, 1500)

# lineArray = [(362, 35), (440, 478)]
# LINE_START = lineArray[0]
# LINE_END = lineArray[1]

#aku-v1
# LINE_START = sv.Point(429, 178)
# LINE_END = sv.Point(539, 474)

# aku-v2
# lineArray = [(362, 38), (463, 477)]
# LINE_START = sv.Point(lineArray[0][0], lineArray[0][1])
# LINE_END = sv.Point(lineArray[1][0], lineArray[1][1])

# people entry
lineArray = [(3, 427), (637, 172)]
LINE_START = sv.Point(lineArray[0][0], lineArray[0][1])
LINE_END = sv.Point(lineArray[1][0], lineArray[1][1])

# uv-v1
# lineArray = [(0, 200), (845, 376)]
# LINE_START = sv.Point(lineArray[0][0], lineArray[0][1])
# LINE_END = sv.Point(lineArray[1][0], lineArray[1][1])

# vg_pop_video1.mp4
# lineArray = [(22, 143), (294, 365)]
# LINE_START = sv.Point(lineArray[0][0], lineArray[0][1])
# LINE_END = sv.Point(lineArray[1][0], lineArray[1][1])
#uv-1
#LINE_START = sv.Point(31, 114)
#LINE_END = sv.Point(719, 213)

#uv-2
# LINE_START = sv.Point(458, 192)
# LINE_END = sv.Point(554, 477)



TARGET_VIDEO_PATH = f"{HOME}/videos/v1_out.mp4"
NEED_BOX_ANNOTATION = True

#TARGET_VIDEO_PATH = "C:\JK\dev\repo\yolov8-native-tracking\videos\new\output_1696750021.mp4"

display.clear_output()
ultralytics.checks()
print("supervision.__version__:", sv.__version__)

model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - car, motorcycle, bus and truck
#selected_classes = [2, 3, 5, 7]
selected_classes = [0]
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
            print(line_points)
            line_points = []

def showOneFrame():
    global frame, line_points
    # create frame generator
    generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    # create instance of BoxAnnotator
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    skip_frames = 1
    # acquire first video frame
    iterator = iter(generator)
    
    # skip frames
    for j in range(0, skip_frames):
        frame = next(iterator)
        
        
    # for i in range(0, 20):
    #     frame = next(iterator)
    #     # write text on frame
    #     cv2.putText(frame, f"Frame : {i*10}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    #     # show the frame and wait for a key press
    #     cv2.imshow('Frame', frame)
    #     # skip 10 frames
    #     for j in range(0, 10):
    #         frame = next(iterator)
        
        
        
    #     # if the `q` key is pressed, break from the loop
    #     if cv2.waitKey(0) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
    #     else:
    #         cv2.destroyAllWindows()
            
       
        
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
    #anotated_frame=box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    #frame = anotated_frame
    #sv.plot_image(anotated_frame, (16,16))
    final_frame = frame
    
    cv2.imshow('Frame', final_frame)
    cv2.line(frame, LINE_START.as_xy_int_tuple(), LINE_END.as_xy_int_tuple(), (203, 255, 0), 2)
    cv2.imshow('Frame', final_frame)
    cv2.setMouseCallback('Frame', draw_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh= 0.25, track_buffer = 30,match_thresh = 0.8,frame_rate =30)

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

# create frame generator
#generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=10)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.25)

# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.25)

frameCount = 0

def scaledowndetections(frame,detections,scale):
    # get bounding boxes from detections and find their center points
    xyxy_objects = detections.xyxy.tolist()
        
    for xyxy in xyxy_objects:
        x1,y1,x2,y2 = xyxy
        # find the center of the box
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        # draw a circle on the center of the box
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        
    
    # for detection in detections:
    #     x1, y1, x2, y2 = detection.bbox
    #     center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

    #     # scale down the detections to 10% of original size towards the center
    #     width = (x2 - x1) * scale
    #     height = (y2 - y1) * scale

    #     # update bounding box coordinates
    #     detection.bbox = [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2]
    
    return detections


def scale_detections(frame,detections, scale_factor):
    scaled_detections = []
    xyxy_objects = detections.xyxy.tolist()
    
    # loop over xyzy_objects
    for xyxy_object in xyxy_objects:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = xyxy_object  # Assuming detection is a tuple or list with format (x1, y1, x2, y2)
        
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        # draw center point
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        
        width, height = x2 - x1, y2 - y1

        # Scale width and height
        new_width = width * scale_factor
        new_height = height * scale_factor

        # Calculate new bounding box coordinates
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2

        # Create a new detection with scaled coordinates
        scaled_detection = (new_x1, new_y1, new_x2, new_y2)

        scaled_detections.append(scaled_detection)
    
    # if len(xyxy_objects) == 0:
    #     return prevStateAllSide
    # strDynamic = ""
    # currentState = [0,0,0,0]
    # detected_object = xyxy_objects[0]
    # boxx1 = detected_object[0]
    # boxy1 = detected_object[1]
    # boxx2 = detected_object[2]
    # boxy2 = detected_object[3]
    
    # for detection in detections:
    #     ultralytics_results.boxes.xyxy.cpu().numpy()
    #     # Extract bounding box coordinates
    #     x1, y1, x2, y2 = detection.boxes.xyxy.cpu().numpy()  # Assuming detection is a tuple or list with format (x1, y1, x2, y2)
        
    #     # Calculate center, width and height
    #     center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    #     width, height = x2 - x1, y2 - y1

    #     # Scale width and height
    #     new_width = width * scale_factor
    #     new_height = height * scale_factor

    #     # Calculate new bounding box coordinates
    #     new_x1 = center_x - new_width / 2
    #     new_y1 = center_y - new_height / 2
    #     new_x2 = center_x + new_width / 2
    #     new_y2 = center_y + new_height / 2

    #     # Create a new detection with scaled coordinates
    #     scaled_detection = (new_x1, new_y1, new_x2, new_y2)

    #     scaled_detections.append(scaled_detection)

    return scaled_detections







# define call back function to be used in video processing
def callback(frame: np.ndarray, index:int) -> np.ndarray:
    global frameCount
    frameCount = frameCount + 1
    print(frameCount)
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, selected_classes)]
    # get bounding boxes from detections and find their center points
    
    # scale down the detections to 10% of original size
    detections = scaledowndetections(frame,detections,0.1)
    #detections = scale_detections(frame,detections, 0.1)
    
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)
    
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    finalFrame = frame
    box_annotated_frame=box_annotator.annotate(scene=frame.copy(),
                                    detections=detections,
                                    labels=labels)
    
    if(NEED_BOX_ANNOTATION):
        finalFrame = box_annotated_frame
        
    # update line counter
    line_zone.trigger(detections)
    print(line_zone.in_count, line_zone.out_count)
    
        
    # return frame with box and line annotated result
    return  line_zone_annotator.annotate(finalFrame, line_counter=line_zone)



#process the whole video
# sv.process_video(
#     source_path = SOURCE_VIDEO_PATH,
#     target_path = TARGET_VIDEO_PATH,
#     callback=callback
# )
def processVideo():
    start_time = time.time()
    source_video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH,stride=1)
        ):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)

    end_time = time.time()

    # Calculate the duration
    duration = end_time - start_time

    print(f"Execution time: {duration:.6f} seconds")
    

#showOneFrame()
processVideo()