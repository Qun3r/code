import cv2
import numpy as np
import sys
import os
import time
from ultralytics import YOLO
import serial

GUI_ENABLED = True
TERMINAL_OUTPUT = True
UART_ENABLED = False

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

BASELINE = 9.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "model_ai.onnx"

LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])

CENTER_MARGIN_CM = 3.0
SHARP_TURN_CM = 8.0

ser = None
if UART_ENABLED:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flush()
        if TERMINAL_OUTPUT:
            print(f"SUCCESS: Serial port {SERIAL_PORT} opened.")
    except Exception as e:
        print(f"ERROR: Could not open serial port: {e}")
else:
    if TERMINAL_OUTPUT:
        print("INFO: UART is disabled in config.")

if not os.path.isfile('stereoMap.xml'):
    print("FATAL ERROR: stereoMap.xml not found!")
    sys.exit()

cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)

mapLx = cv_file.getNode('stereoMapL_x').mat()
mapLy = cv_file.getNode('stereoMapL_y').mat()
mapRx = cv_file.getNode('stereoMapR_x').mat()
mapRy = cv_file.getNode('stereoMapR_y').mat()

projMatrixL_node = cv_file.getNode('projMatrixL')
if projMatrixL_node.empty():
    FOCAL_LENGTH = 500.0
    CX_L = FRAME_WIDTH / 2.0
    if TERMINAL_OUTPUT:
        print("WARNING: projMatrixL not found. Using defaults f=500, cx=320.")
else:
    P1 = projMatrixL_node.mat()
    FOCAL_LENGTH = float(P1[0, 0])
    CX_L = float(P1[0, 2])
    if TERMINAL_OUTPUT:
        print(f"f={FOCAL_LENGTH:.2f}px, cx={CX_L:.2f}px")

cv_file.release()

if mapLx is None or mapRx is None:
    print("FATAL ERROR: Could not load calibration maps!")
    sys.exit()

model = None
if os.path.isfile(MODEL_PATH):
    try:
        model = YOLO(MODEL_PATH, task='detect')
        if TERMINAL_OUTPUT:
            print("✓ AI model loaded")
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        sys.exit()
else:
    print(f"WARNING: Model {MODEL_PATH} not found - AI detection disabled!")

def process_output(cmd_text, distance_cm):
    if TERMINAL_OUTPUT:
        dist_str = f"{distance_cm:.1f}cm" if distance_cm > 0 else "---"
        print(f">> CMD: {cmd_text:12} | DIST: {dist_str}")

    if UART_ENABLED and ser is not None:
        code = b'S'
        if cmd_text == "FORWARD": code = b'F'
        elif cmd_text == "LEFT": code = b'L'
        elif cmd_text == "SHARP LEFT": code = b'l'
        elif cmd_text == "RIGHT": code = b'R'
        elif cmd_text == "SHARP RIGHT": code = b'r'
        elif "STOP" in cmd_text: code = b'S'
        try:
            ser.write(code + b'\n')
        except Exception as e:
            if TERMINAL_OUTPUT:
                print(f"UART WRITE ERROR: {e}")

def detect_ai_left(frame):
    if model is None:
        return None, None

    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return (cx, cy), (x1, y1, x2, y2)
    return None, None

def detect_hsv_smart_right(frame, bbox_left):
    if bbox_left is None:
        return None

    lx1, ly1, lx2, ly2 = bbox_left
    y_start = max(0, ly1 - 40)
    y_end = min(FRAME_HEIGHT, ly2 + 40)
    x_start = 0
    x_end = min(FRAME_WIDTH, lx2 + 20)

    roi = frame[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    (rx, ry), radius = cv2.minEnclosingCircle(c)
    if radius <= 5:
        return None

    return (int(rx) + x_start, int(ry) + y_start)

def compute_X_center_cm(cx_L, cx_R):
    disparity = float(cx_L - cx_R)
    if disparity <= 0:
        return None, None, None

    Z = (FOCAL_LENGTH * BASELINE) / disparity
    X_left = ((float(cx_L) - CX_L) * Z) / FOCAL_LENGTH
    X_center = X_left - (BASELINE / 2.0)
    return X_center, Z, disparity

def get_steering_command_from_X(X_center_cm):
    if abs(X_center_cm) < CENTER_MARGIN_CM:
        return "FORWARD"
    if X_center_cm < -SHARP_TURN_CM:
        return "SHARP LEFT"
    if X_center_cm < 0:
        return "LEFT"
    if X_center_cm > SHARP_TURN_CM:
        return "SHARP RIGHT"
    return "RIGHT"

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)

capL.set(3, FRAME_WIDTH)
capL.set(4, FRAME_HEIGHT)
capR.set(3, FRAME_WIDTH)
capR.set(4, FRAME_HEIGHT)

if not capL.isOpened() or not capR.isOpened():
    print("FATAL ERROR: Cannot open cameras!")
    sys.exit()

if TERMINAL_OUTPUT:
    print("SYSTEM STARTED")

while True:
    t0 = time.time()
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    detL, bboxL = detect_ai_left(rectL)

    cmd = "SEARCHING"
    dist_cm = 0.0
    X_center = None
    disparity = None

    if detL is not None:
        (cx_L, cy_L) = detL
        detR = detect_hsv_smart_right(rectR, bboxL)

        if detR is not None:
            (cx_R, cy_R) = detR
            X_center, Z, disparity = compute_X_center_cm(cx_L, cx_R)
            if Z is not None:
                dist_cm = float(Z)
                if X_center is not None:
                    cmd = get_steering_command_from_X(X_center)
                if dist_cm < 25:
                    cmd = "STOP - OBSTACLE"
        else:
            cmd = "SEARCHING"

        process_output(cmd, dist_cm)
    else:
        process_output("SEARCHING", 0.0)

    if GUI_ENABLED:
        if detL is not None:
            x1, y1, x2, y2 = bboxL
            cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rectL, cmd, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        fps = 1.0 / max(1e-6, (time.time() - t0))
        cv2.putText(rectL, f"FPS:{fps:.1f} f:{FOCAL_LENGTH:.0f}px cx:{CX_L:.0f}px",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if X_center is not None:
            cv2.putText(rectL, f"X_center: {X_center:.1f}cm  Disp:{disparity:.1f}px  Z:{dist_cm:.1f}cm",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow("Left", rectL)
        cv2.imshow("Right", rectR)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capL.release()
capR.release()
cv2.destroyAllWindows()
if UART_ENABLED and ser is not None:
    ser.close()
