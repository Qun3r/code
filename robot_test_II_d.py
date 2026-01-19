import cv2
import numpy as np
import sys
import os
import time
import json
from ultralytics import YOLO
import serial
import csv
import math
from collections import deque

# =========================================================
# TEST D (ETAP II) – AUTONOMICZNY DOJAZD + STOP NA 25 cm
# =========================================================
TEST_D_ENABLED = True

# ile ostatnich poprawnych próbek Z zebrać "tuż przed STOP"
PRESTOP_WINDOW = 10

# zapis CSV
SAVE_CSV = True
CSV_PATH = "testD_autostop_log.csv"

# warm-up (autoekspozycja)
WARMUP_SEC = 2.0

GUI_ENABLED = False
TERMINAL_OUTPUT = True
UART_ENABLED = True

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 9600

BASELINE_CM = 9.516
MODEL_PATH = "model_ai.onnx"

FRAME_W = 640
FRAME_H = 480

LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])

# Twoje progi sterowania (przykład – zostaw jak masz dopracowane)
FORWARD_LEFT_CM  = 15
FORWARD_RIGHT_CM = 5
SHARP_LEFT_CM    = 20
SHARP_RIGHT_CM   = 12

STOP_DIST_CM = 25.0


def load_camera_config(path="camera_config.json"):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Brak {path} (uruchom camera_setup.py)")
    cfg = json.load(open(path, "r", encoding="utf-8"))
    left_idx = int(cfg["left_index"])
    right_idx = int(cfg["right_index"])
    w = int(cfg.get("frame_w", FRAME_W))
    h = int(cfg.get("frame_h", FRAME_H))
    return left_idx, right_idx, w, h


def load_stereo_calibration(xml_path="stereoMap.xml"):
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Brak {xml_path} (uruchom stereo_calibrate.py)")
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)

    mapLx = fs.getNode("stereoMapL_x").mat()
    mapLy = fs.getNode("stereoMapL_y").mat()
    mapRx = fs.getNode("stereoMapR_x").mat()
    mapRy = fs.getNode("stereoMapR_y").mat()

    nodeP1 = fs.getNode("projMatrixL")
    if nodeP1.empty():
        fs.release()
        raise RuntimeError("Brak projMatrixL w stereoMap.xml (kalibracja niepełna).")
    P1 = nodeP1.mat()

    fx = float(P1[0, 0])
    cx = float(P1[0, 2])

    fs.release()

    if mapLx is None or mapRx is None:
        raise RuntimeError("Nie udało się wczytać map rektyfikacji z stereoMap.xml")

    return mapLx, mapLy, mapRx, mapRy, fx, cx


def send_uart_cmd(cmd_text, ser):
    if not UART_ENABLED or ser is None:
        return

    code = b"S"
    if cmd_text == "FORWARD": code = b"F"
    elif cmd_text == "LEFT": code = b"L"
    elif cmd_text == "SHARP LEFT": code = b"l"
    elif cmd_text == "RIGHT": code = b"R"
    elif cmd_text == "SHARP RIGHT": code = b"r"
    elif "STOP" in cmd_text: code = b"S"
    elif cmd_text == "SEARCHING": code = b"S"

    try:
        ser.write(code + b"\n")
    except Exception as e:
        print(f"UART WRITE ERROR: {e}")


def detect_ai_left(model, frame):
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


def detect_hsv_smart_right(frame, bbox_left, frame_w, frame_h):
    if bbox_left is None:
        return None

    lx1, ly1, lx2, ly2 = bbox_left
    y_start = max(0, ly1 - 40)
    y_end = min(frame_h, ly2 + 40)
    x_start = 0
    x_end = min(frame_w, lx2 + 20)

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


def compute_X_center_Z(cx_L, cx_R, fx, cx0, baseline_cm):
    disp = float(cx_L - cx_R)
    if abs(disp) < 1e-6:
        return None, None, None

    Z = (fx * baseline_cm) / abs(disp)
    X_left = ((float(cx_L) - cx0) * Z) / fx
    X_center = X_left - (baseline_cm / 2.0)
    return float(X_center), float(Z), float(disp)


def get_cmd_from_X(Xc):
    # FORWARD jako "okno" przesunięte (Twoje ustawienia)
    if -FORWARD_LEFT_CM <= Xc <= FORWARD_RIGHT_CM:
        return "FORWARD"

    if Xc < -SHARP_LEFT_CM:
        return "SHARP LEFT"
    if Xc < -FORWARD_LEFT_CM:
        return "LEFT"

    if Xc > SHARP_RIGHT_CM:
        return "SHARP RIGHT"
    if Xc > FORWARD_RIGHT_CM:
        return "RIGHT"

    return "FORWARD"


def stats_basic(values):
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return mean, std


def main():
    left_idx, right_idx, fw, fh = load_camera_config()
    global FRAME_W, FRAME_H
    FRAME_W, FRAME_H = fw, fh

    mapLx, mapLy, mapRx, mapRy, fx, cx0 = load_stereo_calibration("stereoMap.xml")

    if TERMINAL_OUTPUT:
        print("=" * 72)
        print("TEST D (Etap II) – Autonomiczny dojazd + STOP")
        print(f"STOP_DIST_CM = {STOP_DIST_CM:.1f} cm | PRESTOP_WINDOW={PRESTOP_WINDOW}")
        print(f"LEFT idx={left_idx} | RIGHT idx={right_idx} | Frame {FRAME_W}x{FRAME_H}")
        print(f"fx={fx:.2f}px cx={cx0:.2f}px | BASELINE={BASELINE_CM} cm")
        print("=" * 72)

    # UART
    ser = None
    if UART_ENABLED:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            ser.flush()
            print(f"SUCCESS: Serial port {SERIAL_PORT} opened.")
        except Exception as e:
            print(f"ERROR: Could not open serial port: {e}")
            ser = None

    # Model
    if not os.path.isfile(MODEL_PATH):
        print(f"FATAL: Model {MODEL_PATH} not found!")
        sys.exit(1)
    model = YOLO(MODEL_PATH, task="detect")
    print("✓ AI model loaded")

    # Cameras
    capL = cv2.VideoCapture(left_idx)
    capR = cv2.VideoCapture(right_idx)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not capL.isOpened() or not capR.isOpened():
        print("FATAL ERROR: Cannot open cameras!")
        sys.exit(1)

    # CSV
    csv_f = None
    csv_w = None
    if SAVE_CSV:
        csv_f = open(CSV_PATH, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["i", "t_sec", "cmd", "ok_full", "Xc_cm", "Z_cm", "disp_px", "cx_L", "cx_R"])

    # Warm-up
    print(f"WARMUP {WARMUP_SEC:.1f}s ...")
    t0 = time.time()
    while time.time() - t0 < WARMUP_SEC:
        capL.read()
        capR.read()

    # Bufor Z przed STOP
    prestop_Z = deque(maxlen=PRESTOP_WINDOW)

    i = 0
    t_start = time.time()
    print("START: robot jedzie – test kończy się na komendzie STOP.")
    try:
        while True:
            i += 1
            loop_t0 = time.time()

            retL, frameL = capL.read()
            retR, frameR = capR.read()
            if not retL or not retR:
                print("CRITICAL: Camera read failed!")
                break

            rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
            rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

            detL, bboxL = detect_ai_left(model, rectL)

            cmd = "SEARCHING"
            ok_full = False
            Xc = None
            Z = None
            disp = None
            cx_L = cx_R = None

            if detL is not None:
                cx_L, cy_L = detL
                detR = detect_hsv_smart_right(rectR, bboxL, FRAME_W, FRAME_H)
                if detR is not None:
                    cx_R, cy_R = detR
                    Xc, Z, disp = compute_X_center_Z(cx_L, cx_R, fx, cx0, BASELINE_CM)
                    if Z is not None and Xc is not None:
                        ok_full = True
                        cmd = get_cmd_from_X(Xc)

                        # logika STOP jak w Twoim kodzie
                        if Z > 0 and Z < STOP_DIST_CM:
                            cmd = "STOP - OBSTACLE"

            # wysyłka UART
            send_uart_cmd(cmd, ser)

            # zapamiętanie Z przed STOP (tylko valid)
            if ok_full and Z is not None and Z > 0:
                prestop_Z.append(float(Z))

            # zapis CSV
            if csv_w is not None:
                csv_w.writerow([
                    i,
                    round(time.time() - t_start, 3),
                    cmd,
                    int(ok_full),
                    None if Xc is None else round(float(Xc), 3),
                    None if Z is None else round(float(Z), 3),
                    None if disp is None else round(float(disp), 3),
                    cx_L, cx_R
                ])

            if TERMINAL_OUTPUT:
                if ok_full:
                    print(f"i={i:4d} cmd={cmd:14} Z={Z:6.1f}cm Xc={Xc:6.2f}cm")
                else:
                    print(f"i={i:4d} cmd={cmd:14} (no full det)")

            # stop warunek testu
            if cmd == "STOP - OBSTACLE":
                # dodatkowo wyślij STOP jeszcze raz dla pewności
                send_uart_cmd("STOP - OBSTACLE", ser)

                mean_std = stats_basic(list(prestop_Z))
                print("-" * 72)
                print("[TEST D] STOP triggered.")
                print(f"[TEST D] Last {len(prestop_Z)} valid Z samples before STOP: {list(prestop_Z)}")
                if mean_std is not None:
                    meanZ, stdZ = mean_std
                    print(f"[TEST D] preSTOP meanZ={meanZ:.1f}cm | stdZ={stdZ:.2f}cm")
                print("[TEST D] Teraz zmierz taśmą Z_real i policz błąd zatrzymania:")
                print(f"         e_stop = Z_real - {STOP_DIST_CM:.1f}cm")
                print("-" * 72)
                break

            _ = loop_t0  # dla IDE

    except KeyboardInterrupt:
        print("Interrupted.")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    if ser is not None:
        ser.close()
    if csv_f is not None:
        csv_f.close()
        print(f"[TEST D] CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
