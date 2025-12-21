import cv2
import numpy as np
import sys
import os
import time
from ultralytics import YOLO
import serial

# ================= 1. KONFIGURACJA TRYBU PRACY =================

# --- 3 ZMIENNE KONTROLNE ---

# 1. Czy wyświetlać okienka z obrazem? (GUI)q
GUI_ENABLED = True

# 2. Czy wypisywać komendy w terminalu/konsoli?
TERMINAL_OUTPUT = False

# 3. Czy wysyłać sygnały do Nexteer przez UART?
UART_ENABLED = False

# ================= 2. KONFIGURACJA SPRZĘTOWA =================

# Parametry komunikacji UART
SERIAL_PORT = '/dev/ttyS0'
BAUD_RATE = 9600

# Parametry wizyjne i fizyczne
BASELINE = 9.5        # ZMIENIONO: Odległość między kamerami [cm]
FOCAL_LENGTH = 500    # [px]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "model_ai.onnx" # ZMIENIONO: Ścieżka do wytrenowanego modelu ONNX

# Parametry detekcji
LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])
CENTER_MARGIN = 80
SHARP_TURN_LIMIT = 120

# ================= 3. INICJALIZACJA SYSTEMU =================

# A. Inicjalizacja UART (tylko jeśli włączony)
ser = None
if UART_ENABLED:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flush()
        if TERMINAL_OUTPUT: print(f"SUCCESS: Serial port {SERIAL_PORT} opened.")
    except Exception as e:
        print(f"ERROR: Could not open serial port: {e}")
else:
    if TERMINAL_OUTPUT: print("INFO: UART is disabled in config.")

# B. Wczytanie kalibracji
if not os.path.isfile('stereoMap.xml'):
    print("FATAL ERROR: stereoMap.xml not found!")
    sys.exit()

if TERMINAL_OUTPUT: print("Loading calibration maps...")
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)
mapLx = cv_file.getNode('stereoMapL_x').mat()
mapLy = cv_file.getNode('stereoMapL_y').mat()
mapRx = cv_file.getNode('stereoMapR_x').mat()
mapRy = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

# C. Wczytanie modelu AI
if TERMINAL_OUTPUT: print(f"Loading AI model: {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH, task='detect')
except Exception as e:
    print(f"FATAL ERROR loading model: {e}")
    sys.exit()

# ================= 4. FUNKCJE LOGICZNE =================

def process_output(cmd_text, distance_cm):
    """
    Zarządza wysyłaniem danych (Terminal i UART) w jednym miejscu.
    """
    # 1. Wypisywanie w terminalu
    if TERMINAL_OUTPUT:
        dist_str = f"{distance_cm:.1f}cm" if distance_cm > 0 else "---"
        print(f">> CMD: {cmd_text:12} | DIST: {dist_str}")

    # 2. Wysyłanie przez UART
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
            if TERMINAL_OUTPUT: print(f"UART WRITE ERROR: {e}")

def detect_ai_left(frame):
    """AI Detection on Left Camera"""
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
    """HSV Helper on Right Camera"""
    if bbox_left is None: return None, None
    lx1, ly1, lx2, ly2 = bbox_left
    y_start = max(0, ly1 - 40); y_end = min(FRAME_HEIGHT, ly2 + 40)
    x_start = 0; x_end = min(FRAME_WIDTH, lx2 + 20)

    roi_frame = frame[y_start:y_end, x_start:x_end]
    if roi_frame.size == 0: return None, None

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((rx_local, ry_local), radius) = cv2.minEnclosingCircle(c)
        if radius > 5:
            return int(rx_local) + x_start, int(ry_local) + y_start
    return None, None

def get_steering_command(x_pos):
    center = FRAME_WIDTH // 2
    if abs(x_pos - center) < CENTER_MARGIN: return "FORWARD"
    if x_pos < SHARP_TURN_LIMIT: return "SHARP LEFT"
    elif x_pos < center: return "LEFT"
    elif x_pos > FRAME_WIDTH - SHARP_TURN_LIMIT: return "SHARP RIGHT"
    else: return "RIGHT"

# ================= 5. PĘTLA GŁÓWNA =================

if TERMINAL_OUTPUT: print("Initializing cameras...")
capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)
capL.set(3, FRAME_WIDTH); capL.set(4, FRAME_HEIGHT)
capR.set(3, FRAME_WIDTH); capR.set(4, FRAME_HEIGHT)

if TERMINAL_OUTPUT: print("SYSTEM STARTED. Main loop running...")

while True:
    loop_start = time.time()

    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        if TERMINAL_OUTPUT: print("CRITICAL: Camera read failed!")
        break

    # 1. REKTYFIKACJA
    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    # 2. DETEKCJA I LOGIKA
    cx_L, cy_L, rad_L, bbox_L = detect_ai_left(rectL)

    command = "SEARCHING"
    dist_cm = 0.0

    if cx_L is not None:
        # A. Ustalenie kierunku
        command = get_steering_command(cx_L)

        # B. Ustalenie odległości
        cx_R, cy_R = detect_hsv_smart_right(rectR, bbox_L)

        if cx_R is not None:
            disparity = cx_L - cx_R
            if disparity > 0:
                # WZÓR NA GŁĘBIĘ
                depth_cm = (FOCAL_LENGTH * BASELINE) / disparity
                dist_cm = depth_cm

                # C. Hamowanie awaryjne
                if dist_cm < 25:
                    command = "STOP - OBSTACLE"

    # 3. OBSŁUGA WYJŚCIA (TERMINAL + UART)
    if cx_L is not None:
        process_output(command, dist_cm)
    else:
        if TERMINAL_OUTPUT: process_output("SEARCHING", dist_cm)


    # 4. OBSŁUGA GRAFIKI (GUI)
    if GUI_ENABLED:
        # Rysowanie linii stref
        cv2.line(rectL, (SHARP_TURN_LIMIT, 0), (SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH - SHARP_TURN_LIMIT, 0), (FRAME_WIDTH - SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH // 2 - CENTER_MARGIN, 0), (FRAME_WIDTH // 2 - CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH // 2 + CENTER_MARGIN, 0), (FRAME_WIDTH // 2 + CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)

        # Rysowanie wykryć
        if cx_L is not None:
            x1, y1, x2, y2 = bbox_L
            cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rectL, f"{command}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if cx_R is not None:
                cv2.circle(rectR, (cx_R, cy_R), 8, (0, 0, 255), 2)
                if dist_cm > 0:
                    cv2.putText(rectL, f"Dist: {dist_cm:.1f}cm", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS licznik
        fps = 1.0 / (time.time() - loop_start)
        cv2.putText(rectL, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Wyświetlanie okien
        cv2.imshow("Left (AI Control)", rectL)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capL.release()
capR.release()
cv2.destroyAllWindows()