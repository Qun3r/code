import cv2
import numpy as np
import sys
import os
import time
from ultralytics import YOLO
import serial

# ================= 1. KONFIGURACJA TRYBU PRACY =================

GUI_ENABLED = True
TERMINAL_OUTPUT = True
UART_ENABLED = False  # ZMIEŃ NA True gdy podłączysz UART

# ================= 2. KONFIGURACJA SPRZĘTOWA =================

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

# Parametry wizyjne i fizyczne
BASELINE = 9.5        # Odległość między kamerami [cm]
FOCAL_LENGTH = 500    # [px]
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "model_ai.onnx"

# Parametry detekcji
LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])
CENTER_MARGIN = 80
SHARP_TURN_LIMIT = 120

# ================= 3. INICJALIZACJA SYSTEMU =================

# A. Inicjalizacja UART
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
    print("Najpierw uruchom:")
    print("1. calibration_images_FIXED.py (zrób ~20 zdjęć)")
    print("2. stereovision_calibration.py")
    print("3. verify_calibration.py (sprawdź jakość)")
    sys.exit()

if TERMINAL_OUTPUT: print("Loading calibration maps...")
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)

# Mapy kalibracyjne zgodne z kolejnością kamer
mapLx = cv_file.getNode('stereoMapL_x').mat()
mapLy = cv_file.getNode('stereoMapL_y').mat()
mapRx = cv_file.getNode('stereoMapR_x').mat()
mapRy = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

if mapLx is None or mapRx is None:
    print("FATAL ERROR: Could not load calibration maps!")
    sys.exit()

if TERMINAL_OUTPUT: print("✓ Calibration maps loaded successfully")

# C. Wczytanie modelu AI
if os.path.isfile(MODEL_PATH):
    if TERMINAL_OUTPUT: print(f"Loading AI model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task='detect')
        if TERMINAL_OUTPUT: print("✓ AI model loaded")
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        sys.exit()
else:
    print(f"WARNING: Model {MODEL_PATH} not found - AI detection disabled!")
    model = None

# ================= 4. FUNKCJE LOGICZNE =================

def process_output(cmd_text, distance_cm):
    """Zarządza wysyłaniem danych (Terminal i UART)"""
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
            if TERMINAL_OUTPUT: print(f"UART WRITE ERROR: {e}")

def detect_ai_left(frame):
    """AI Detection on Left Camera"""
    if model is None:
        return None, None, None, None
    
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
            return int(rx_local) + x_start, int(ry_local) + y_start
    return None, None

def get_steering_command(x_pos):
    """Determine steering based on X position"""
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

# ================= 5. PĘTLA GŁÓWNA =================

if TERMINAL_OUTPUT: 
    print("=" * 60)
    print("Initializing cameras...")
    print("LEWA kamera = index 0 (lewa strona robota)")
    print("PRAWA kamera = index 2 (prawa strona robota)")
    print("=" * 60)

# POPRAWIONA KOLEJNOŚĆ zgodna z fizycznym montażem
capL = cv2.VideoCapture(0)  # LEWA
capR = cv2.VideoCapture(2)  # PRAWA

capL.set(3, FRAME_WIDTH)
capL.set(4, FRAME_HEIGHT)
capR.set(3, FRAME_WIDTH)
capR.set(4, FRAME_HEIGHT)

if not capL.isOpened() or not capR.isOpened():
    print("FATAL ERROR: Cannot open cameras!")
    sys.exit()

if TERMINAL_OUTPUT: 
    print("✓ Cameras opened successfully")
    print("=" * 60)
    print("SYSTEM STARTED. Main loop running...")
    print("Press 'q' to quit")
    print("=" * 60)

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
                depth_cm = (FOCAL_LENGTH * BASELINE) / disparity
                dist_cm = depth_cm

                # C. Hamowanie awaryjne
                if dist_cm < 25:
                    command = "STOP - OBSTACLE"

    # 3. OBSŁUGA WYJŚCIA
    if cx_L is not None:
        process_output(command, dist_cm)
    else:
        if TERMINAL_OUTPUT: 
            process_output("SEARCHING", dist_cm)

    # 4. OBSŁUGA GRAFIKI
    if GUI_ENABLED:
        # Rysowanie linii stref
        cv2.line(rectL, (SHARP_TURN_LIMIT, 0), (SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH - SHARP_TURN_LIMIT, 0), (FRAME_WIDTH - SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH // 2 - CENTER_MARGIN, 0), (FRAME_WIDTH // 2 - CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH // 2 + CENTER_MARGIN, 0), (FRAME_WIDTH // 2 + CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)

        # Etykiety kamer
        cv2.putText(rectL, "LEWA (index 0)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(rectR, "PRAWA (index 2)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
        fps = 1.0 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
        cv2.putText(rectL, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Wyświetlanie okien
        cv2.imshow("Left (AI Control)", rectL)
        cv2.imshow("Right (Helper)", rectR)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if TERMINAL_OUTPUT: print("\nSYSTEM SHUTDOWN")

capL.release()
capR.release()
cv2.destroyAllWindows()

if UART_ENABLED and ser is not None:
    ser.close()
