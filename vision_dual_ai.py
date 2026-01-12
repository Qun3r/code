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

# KLUCZOWE: Offset kamery względem centrum robota
# Lewa kamera jest przesunięta o połowę baseline w lewo od centrum robota
CAMERA_OFFSET_CM = BASELINE / 2  # 4.75 cm

# Parametry detekcji
CENTER_MARGIN = 80
SHARP_TURN_LIMIT = 120
MIN_CONFIDENCE = 0.5  # Minimalny confidence score dla detekcji

# Parametry dopasowania obiektów między kamerami
MAX_Y_DIFF = 50       # Maksymalna różnica w Y między detekcjami (px)
MAX_SIZE_DIFF = 0.5   # Maksymalna różnica w rozmiarze (procent)

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
    print("2. stereovision_calibration_IMPROVED.py")
    print("3. verify_calibration.py (sprawdź jakość)")
    sys.exit()

if TERMINAL_OUTPUT: print("Loading calibration maps...")
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)

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
    print(f"FATAL ERROR: Model {MODEL_PATH} not found!")
    sys.exit()

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

def detect_objects_ai(frame):
    """
    Wykrywa wszystkie obiekty na obrazie używając AI
    Zwraca listę detekcji: [(cx, cy, confidence, bbox), ...]
    """
    detections = []
    results = model(frame, imgsz=320, conf=MIN_CONFIDENCE, verbose=False)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            confidence = float(box.conf[0])
            width = x2 - x1
            height = y2 - y1
            
            detections.append({
                'cx': cx,
                'cy': cy,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'width': width,
                'height': height
            })
    
    # Sortuj po confidence (najlepsze pierwsze)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections

def match_detections(det_left, det_right):
    """
    Dopasowuje detekcje z lewej i prawej kamery
    Zwraca parę (det_L, det_R) lub (None, None)
    """
    if not det_left or not det_right:
        return None, None
    
    # Dla każdej detekcji z lewej kamery, szukaj najlepszego dopasowania w prawej
    best_match = None
    best_score = float('inf')
    
    for det_l in det_left:
        for det_r in det_right:
            # Sprawdź czy detekcje są na podobnej wysokości (Y)
            y_diff = abs(det_l['cy'] - det_r['cy'])
            if y_diff > MAX_Y_DIFF:
                continue
            
            # Sprawdź czy rozmiary są podobne
            size_l = det_l['width'] * det_l['height']
            size_r = det_r['width'] * det_r['height']
            size_ratio = min(size_l, size_r) / max(size_l, size_r)
            
            if size_ratio < (1.0 - MAX_SIZE_DIFF):
                continue
            
            # Sprawdź czy det_r jest na lewo od det_l (dla stereo)
            # W układzie stereo: obiekt bliżej = większa dyspersja = det_r bardziej w lewo
            if det_r['cx'] > det_l['cx']:
                continue
            
            # Oblicz score dopasowania (mniejszy = lepszy)
            score = y_diff + (1.0 - size_ratio) * 100
            
            if score < best_score:
                best_score = score
                best_match = (det_l, det_r)
    
    return best_match if best_match else (None, None)

def calculate_distance(det_left, det_right):
    """
    Oblicza odległość na podstawie dyspersji
    Zwraca odległość w cm
    """
    disparity = det_left['cx'] - det_right['cx']
    
    if disparity <= 0:
        return 0.0
    
    # Wzór na głębię ze stereo: depth = (focal_length * baseline) / disparity
    depth_cm = (FOCAL_LENGTH * BASELINE) / disparity
    
    return depth_cm

def calculate_robot_center_position(det_left, distance_cm):
    """
    Oblicza pozycję obiektu względem CENTRUM ROBOTA (nie lewej kamery!)
    
    Parametry:
    - det_left: detekcja z lewej kamery
    - distance_cm: odległość do obiektu
    
    Zwraca:
    - x_position: pozycja X względem centrum robota w pikselach
    """
    if distance_cm <= 0:
        # Jeśli nie mamy odległości, nie możemy dokładnie skorygować
        # Zakładamy średnią odległość dla przybliżenia
        distance_cm = 100  # cm (wartość domyślna)
    
    # 1. Pozycja X obiektu w lewej kamerze (od lewej krawędzi obrazu)
    x_in_left_camera = det_left['cx']
    
    # 2. Oblicz przesunięcie w pikselach spowodowane offsetem kamery
    # offset_px = (camera_offset_cm / distance_cm) * focal_length
    offset_px = (CAMERA_OFFSET_CM / distance_cm) * FOCAL_LENGTH
    
    # 3. Skoryguj pozycję - przesuń "wirtualne centrum" w prawo
    # Im bliżej obiekt, tym większe przesunięcie
    x_robot_center = x_in_left_camera + offset_px
    
    return x_robot_center

def get_steering_command(x_pos):
    """
    Określa kierunek skrętu na podstawie pozycji X
    UWAGA: x_pos powinno być już skorygowane względem centrum robota!
    """
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
    print("DUAL AI SYSTEM STARTED")
    print("Press 'q' to quit")
    print("=" * 60)

# Statystyki
frame_count = 0
total_detections_L = 0
total_detections_R = 0
total_matches = 0

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

    # 2. DETEKCJA AI W OBUDU KAMERACH
    detections_L = detect_objects_ai(rectL)
    detections_R = detect_objects_ai(rectR)
    
    total_detections_L += len(detections_L)
    total_detections_R += len(detections_R)

    # 3. DOPASOWANIE DETEKCJI
    det_L, det_R = match_detections(detections_L, detections_R)

    command = "SEARCHING"
    dist_cm = 0.0
    x_robot_center = None

    if det_L is not None and det_R is not None:
        total_matches += 1
        
        # A. Obliczenie odległości
        dist_cm = calculate_distance(det_L, det_R)

        # B. KLUCZOWE: Oblicz pozycję względem CENTRUM ROBOTA
        x_robot_center = calculate_robot_center_position(det_L, dist_cm)
        
        # C. Ustalenie kierunku (bazując na skorygowanej pozycji!)
        command = get_steering_command(x_robot_center)

        # D. Hamowanie awaryjne
        if dist_cm > 0 and dist_cm < 25:
            command = "STOP - OBSTACLE"
    
    # 4. OBSŁUGA WYJŚCIA
    process_output(command, dist_cm)

    # 5. OBSŁUGA GRAFIKI
    if GUI_ENABLED:
        # Rysowanie linii stref na lewej kamerze
        # UWAGA: Te linie reprezentują strefy dla LEWEJ KAMERY, nie centrum robota
        cv2.line(rectL, (SHARP_TURN_LIMIT, 0), (SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH - SHARP_TURN_LIMIT, 0), (FRAME_WIDTH - SHARP_TURN_LIMIT, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH // 2 - CENTER_MARGIN, 0), (FRAME_WIDTH // 2 - CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)
        cv2.line(rectL, (FRAME_WIDTH // 2 + CENTER_MARGIN, 0), (FRAME_WIDTH // 2 + CENTER_MARGIN, FRAME_HEIGHT), (0, 0, 255), 2)
        
        # Linia pokazująca centrum obrazu lewej kamery
        cv2.line(rectL, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (255, 0, 255), 1)

        # Etykiety kamer
        cv2.putText(rectL, "LEWA (AI) - Robot Center Corrected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(rectR, "PRAWA (AI)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Rysowanie WSZYSTKICH detekcji (półprzezroczyste dla niezmatched)
        for det in detections_L:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) if det == det_L else (100, 100, 100)
            thickness = 2 if det == det_L else 1
            cv2.rectangle(rectL, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(rectL, f"{det['confidence']:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for det in detections_R:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 0, 255) if det == det_R else (100, 100, 100)
            thickness = 2 if det == det_R else 1
            cv2.rectangle(rectR, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(rectR, f"{det['confidence']:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Informacje o matched detection
        if det_L is not None and det_R is not None:
            # Komenda na lewej kamerze
            x1, y1, x2, y2 = det_L['bbox']
            cv2.putText(rectL, f"{command}", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Wizualizacja skorygowanej pozycji
            if x_robot_center is not None:
                # Narysuj krzyżyk pokazujący gdzie robot "widzi" obiekt względem swojego centrum
                x_corrected = int(x_robot_center)
                if 0 <= x_corrected < FRAME_WIDTH:
                    cv2.line(rectL, (x_corrected, 0), (x_corrected, FRAME_HEIGHT), (0, 255, 255), 2)
                    cv2.putText(rectL, "Robot Center View", (x_corrected + 5, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Pokaż offset w pikselach
                offset_px = x_robot_center - det_L['cx']
                cv2.putText(rectL, f"Offset: {offset_px:.1f}px", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Odległość
            if dist_cm > 0:
                cv2.putText(rectL, f"Dist: {dist_cm:.1f}cm", (10, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Dyspersja (do debugowania)
                disparity = det_L['cx'] - det_R['cx']
                cv2.putText(rectL, f"Disp: {disparity}px", (10, 420), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # FPS i statystyki
        fps = 1.0 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
        cv2.putText(rectL, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        frame_count += 1
        if frame_count % 30 == 0 and TERMINAL_OUTPUT:
            match_rate = (total_matches / frame_count * 100) if frame_count > 0 else 0
            print(f"Stats: L={len(detections_L)} R={len(detections_R)} Matched={total_matches} Rate={match_rate:.1f}%")

        # Wyświetlanie okien
        cv2.imshow("Left Camera (AI Control)", rectL)
        cv2.imshow("Right Camera (AI Helper)", rectR)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Podsumowanie
if TERMINAL_OUTPUT: 
    print("\n" + "=" * 60)
    print("SYSTEM SHUTDOWN")
    print(f"Total frames: {frame_count}")
    print(f"Avg detections L: {total_detections_L/frame_count:.1f}")
    print(f"Avg detections R: {total_detections_R/frame_count:.1f}")
    print(f"Total matches: {total_matches}")
    print(f"Match rate: {total_matches/frame_count*100:.1f}%")
    print("=" * 60)

capL.release()
capR.release()
cv2.destroyAllWindows()

if UART_ENABLED and ser is not None:
    ser.close()
