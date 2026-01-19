import cv2
import numpy as np
import sys
import os
import time
import json
from ultralytics import YOLO
import serial
import csv

# =========================================================
# TEST C (ETAP II) – AUTOMATYZACJA
# Ciągłość detekcji w ruchu: TP/FN + Recall + statystyki
# =========================================================
TEST_C_ENABLED = True

# ile iteracji testu (1 = "pojedyncza iteracja do udokumentowania")
TEST_C_MAX_ITERS = 300

# co ile iteracji wypisywać raport pośredni
TEST_C_PRINT_EVERY = 100

# zapis logu do CSV (polecam zostawić True)
TEST_C_SAVE_CSV = True
TEST_C_CSV_PATH = "testC_log.csv"

# krótki warm-up po starcie (żeby kamera/autoekspozycja się ustabilizowała)
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

STOP_DIST_CM = 25.0

# strefy sterowania – te same co w Twoim pliku testy.py
FORWARD_LEFT_CM  = 10
FORWARD_RIGHT_CM = 3
SHARP_LEFT_CM    = 18
SHARP_RIGHT_CM   = 12


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


def process_output(cmd_text, distance_cm, ser=None):
    if TERMINAL_OUTPUT:
        dist_str = f"{distance_cm:.1f}cm" if distance_cm > 0 else "---"
        print(f">> CMD: {cmd_text:14} | DIST: {dist_str}")

    if UART_ENABLED and ser is not None:
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
            if TERMINAL_OUTPUT:
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


def compute_X_center_cm(cx_L, cx_R, fx, cx0, baseline_cm):
    disp = float(cx_L - cx_R)
    if abs(disp) < 1e-6:
        return None, None, None

    Z = (fx * baseline_cm) / abs(disp)
    X_left = ((float(cx_L) - cx0) * Z) / fx
    X_center = X_left - (baseline_cm / 2.0)
    return X_center, Z, disp


def get_steering_command_from_X(X_center_cm):
    # FORWARD w zakresie przesuniętym w prawo
    if -FORWARD_LEFT_CM <= X_center_cm <= FORWARD_RIGHT_CM:
        return "FORWARD"

    if X_center_cm < -SHARP_LEFT_CM:
        return "SHARP LEFT"
    if X_center_cm < -FORWARD_LEFT_CM:
        return "LEFT"

    if X_center_cm > SHARP_RIGHT_CM:
        return "SHARP RIGHT"
    if X_center_cm > FORWARD_RIGHT_CM:
        return "RIGHT"

    return "FORWARD"


def print_testC_stats(N, TP, FN, cmd_counts, switches, fps_mean):
    denom = TP + FN
    recall = (TP / denom) if denom > 0 else 0.0

    def pct(x): 
        return 100.0 * x / N if N > 0 else 0.0

    print("=" * 62)
    print(f"[TEST C] N={N} | TP={TP} | FN={FN} | Recall={recall:.3f}")
    print(f"[TEST C] FPS_mean={fps_mean:.2f} | Command switches={switches}")
    print("[TEST C] CMD distribution:")
    for k in ["FORWARD", "LEFT", "RIGHT", "SHARP LEFT", "SHARP RIGHT", "STOP - OBSTACLE", "SEARCHING"]:
        v = cmd_counts.get(k, 0)
        print(f"  - {k:14}: {v:5d}  ({pct(v):5.1f}%)")
    print("=" * 62)


def main():
    try:
        left_idx, right_idx, fw, fh = load_camera_config()
    except Exception as e:
        print("FATAL:", e)
        sys.exit(1)

    global FRAME_W, FRAME_H
    FRAME_W, FRAME_H = fw, fh

    try:
        mapLx, mapLy, mapRx, mapRy, fx, cx0 = load_stereo_calibration("stereoMap.xml")
    except Exception as e:
        print("FATAL:", e)
        sys.exit(1)

    if TERMINAL_OUTPUT:
        print("=" * 60)
        print("CONFIG:")
        print(f"LEFT camera index : {left_idx}")
        print(f"RIGHT camera index: {right_idx}")
        print(f"Frame: {FRAME_W}x{FRAME_H}")
        print(f"fx={fx:.2f}px, cx={cx0:.2f}px (from stereoMap.xml)")
        print(f"BASELINE={BASELINE_CM} cm")
        print(f"TEST C: max_iters={TEST_C_MAX_ITERS} print_every={TEST_C_PRINT_EVERY}")
        print("=" * 60)

    ser = None
    if UART_ENABLED:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            ser.flush()
            if TERMINAL_OUTPUT:
                print(f"SUCCESS: Serial port {SERIAL_PORT} opened.")
        except Exception as e:
            print(f"ERROR: Could not open serial port: {e}")
            ser = None

    model = None
    if os.path.isfile(MODEL_PATH):
        try:
            model = YOLO(MODEL_PATH, task="detect")
            if TERMINAL_OUTPUT:
                print("✓ AI model loaded")
        except Exception as e:
            print("FATAL ERROR loading model:", e)
            sys.exit(1)
    else:
        print(f"WARNING: Model {MODEL_PATH} not found - AI detection disabled!")

    capL = cv2.VideoCapture(left_idx)
    capR = cv2.VideoCapture(right_idx)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not capL.isOpened() or not capR.isOpened():
        print("FATAL ERROR: Cannot open cameras!")
        sys.exit(1)

    # CSV log
    csv_f = None
    csv_w = None
    if TEST_C_SAVE_CSV:
        csv_f = open(TEST_C_CSV_PATH, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow([
            "i", "t_sec", "ok_full", "cmd",
            "cx_L", "cy_L", "cx_R", "cy_R",
            "disp_px", "Xc_cm", "Z_cm"
        ])

    if TERMINAL_OUTPUT:
        print("SYSTEM STARTED - Test C running... (CTRL+C to stop)")
        print(f"WARMUP {WARMUP_SEC:.1f}s ...")

    t_start = time.time()
    while time.time() - t_start < WARMUP_SEC:
        capL.read()
        capR.read()

    # ---- liczniki Test C ----
    N = 0
    TP = 0
    FN = 0

    cmd_counts = {}
    switches = 0
    prev_cmd = None

    fps_sum = 0.0
    fps_cnt = 0

    try:
        while True:
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
            dist_cm = 0.0
            X_center = None
            disp = None
            cx_L = cy_L = None
            cx_R = cy_R = None

            detR = None
            if detL is not None:
                cx_L, cy_L = detL
                detR = detect_hsv_smart_right(rectR, bboxL, FRAME_W, FRAME_H)

                if detR is not None:
                    cx_R, cy_R = detR
                    X_center, Z, disp = compute_X_center_cm(cx_L, cx_R, fx, cx0, BASELINE_CM)
                    if Z is not None:
                        dist_cm = float(Z)

                    if X_center is not None:
                        cmd = get_steering_command_from_X(X_center)

                    if dist_cm > 0 and dist_cm < STOP_DIST_CM:
                        cmd = "STOP - OBSTACLE"
                else:
                    cmd = "SEARCHING"

            # --- TEST C definicja sukcesu: komplet detekcji YOLO+HSV ---
            ok_full = (detL is not None) and (detR is not None)

            if TEST_C_ENABLED:
                N += 1
                if ok_full:
                    TP += 1
                else:
                    FN += 1

                cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1

                if prev_cmd is not None and cmd != prev_cmd:
                    switches += 1
                prev_cmd = cmd

                fps = 1.0 / max(1e-6, (time.time() - loop_t0))
                fps_sum += fps
                fps_cnt += 1

                # zapis CSV
                if csv_w is not None:
                    csv_w.writerow([
                        N,
                        round(time.time() - t_start, 3),
                        int(ok_full),
                        cmd,
                        cx_L, cy_L, cx_R, cy_R,
                        None if disp is None else round(float(disp), 3),
                        None if X_center is None else round(float(X_center), 3),
                        None if dist_cm <= 0 else round(float(dist_cm), 3)
                    ])

                # raport pośredni
                if (N % TEST_C_PRINT_EVERY) == 0:
                    fps_mean = fps_sum / max(1, fps_cnt)
                    print_testC_stats(N, TP, FN, cmd_counts, switches, fps_mean)

                # stop po osiągnięciu limitu
                if TEST_C_MAX_ITERS > 0 and N >= TEST_C_MAX_ITERS:
                    fps_mean = fps_sum / max(1, fps_cnt)
                    print_testC_stats(N, TP, FN, cmd_counts, switches, fps_mean)
                    print("[TEST C] Done.")
                    break

            # normalne sterowanie (w trakcie testu robot dalej jeździ)
            process_output(cmd, dist_cm, ser)

    except KeyboardInterrupt:
        fps_mean = fps_sum / max(1, fps_cnt)
        print_testC_stats(N, TP, FN, cmd_counts, switches, fps_mean)
        print("[TEST C] Interrupted by user.")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    if UART_ENABLED and ser is not None:
        ser.close()
    if csv_f is not None:
        csv_f.close()
        print(f"[TEST C] CSV saved to: {TEST_C_CSV_PATH}")


if __name__ == "__main__":
    main()
