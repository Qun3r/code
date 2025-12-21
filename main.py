import cv2
import numpy as np
import sys
import os
from ultralytics import YOLO

BASELINE = 9.5
FOCAL_LENGTH = 500
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "model_ai.onnx"

LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])

CENTER_MARGIN = 80
SHARP_TURN_LIMIT = 120

if not os.path.isfile('stereoMap.xml'):
    print("ERROR: stereoMap.xml not found!")
    sys.exit()

print("Loading calibration maps...")
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)
mapLx = cv_file.getNode('stereoMapL_x').mat()
mapLy = cv_file.getNode('stereoMapL_y').mat()
mapRx = cv_file.getNode('stereoMapR_x').mat()
mapRy = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

print(f"Loading AI model from {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH, task='detect')
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit()

def detect_ai_left(frame):
    results = model(frame, imgsz=320, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            radius = max(x2 - x1, y2 - y1) // 2
            return cx, cy, radius, (x1, y1, x2, y2)

    return None, None, None, None

def detect_hsv_smart_right(frame, bbox_left):
    if bbox_left is None:
        return None, None

    lx1, ly1, lx2, ly2 = bbox_left

    y_start = max(0, ly1 - 40)
    y_end = min(FRAME_HEIGHT, ly2 + 40)
    x_start = 0
    x_end = min(FRAME_WIDTH, lx2 + 20)

    roi_frame = frame[y_start:y_end, x_start:x_end]

    if roi_frame.size == 0:
        return None, None

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((rx_local, ry_local), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            global_x = int(rx_local) + x_start
            global_y = int(ry_local) + y_start
            return global_x, global_y

    return None, None

def get_steering_command(x_pos):
    center = FRAME_WIDTH // 2

    if abs(x_pos - center) < CENTER_MARGIN:
        return "FORWARD"

    if x_pos < SHARP_TURN_LIMIT:
        return "SHARP LEFT"
    elif x_pos < center:
        return "LEFT"
    elif x_pos > FRAME_WIDTH - SHARP_TURN_LIMIT:
        return "SHARP RIGHT"
    else:
        return "RIGHT"

print("Initializing cameras...")
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)

capL.set(3, FRAME_WIDTH); capL.set(4, FRAME_HEIGHT)
capR.set(3, FRAME_WIDTH); capR.set(4, FRAME_HEIGHT)

print("System started. Press 'q' to exit.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Camera error!")
        break

   
    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    cx_L, cy_L, rad_L, bbox_L = detect_ai_left(rectL)

   
    cv2.line(rectL, (SHARP_TURN_LIMIT, 0), (SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
    cv2.line(rectL, (FRAME_WIDTH - SHARP_TURN_LIMIT, 0), (FRAME_WIDTH - SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
    cv2.line(rectL, (FRAME_WIDTH // 2 - CENTER_MARGIN, 0), (FRAME_WIDTH // 2 - CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)
    cv2.line(rectL, (FRAME_WIDTH // 2 + CENTER_MARGIN, 0), (FRAME_WIDTH // 2 + CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)



    command = "SEARCHING..."
    distance_message = ""

    if cx_L is not None:
        x1, y1, x2, y2 = bbox_L
        cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rectL, "AI", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        command = get_steering_command(cx_L)

        cx_R, cy_R = detect_hsv_smart_right(rectR, bbox_L)

        if cx_R is not None:
            cv2.circle(rectR, (cx_R, cy_R), 10, (0, 0, 255), 2)

            disparity = cx_L - cx_R

            if disparity > 0:
                depth_cm = (FOCAL_LENGTH * BASELINE) / disparity
                distance_message = f"Dist: {depth_cm:.1f} cm"

                if depth_cm < 25:
                    command = "STOP - OBSTACLE"

            if abs(cy_L - cy_R) > 30:
                cv2.putText(rectL, "WARN: Y-ALIGN ERROR", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.putText(rectL, f"CMD: {command}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    cv2.putText(rectL, distance_message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Left Camera (Control)", rectL)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()