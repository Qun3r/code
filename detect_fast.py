import cv2
import numpy as np
import sys
import os
import time
import json
from ultralytics import YOLO
import serial

# =========================
# FAST / DEBUG ustawienia
# =========================
GUI_ENABLED = False
TERMINAL_OUTPUT = False
UART_ENABLED = True

# Największy zysk FPS:
RECTIFY_ENABLED = True   # False = szybki podgląd/sterowanie, True = poprawna Z z dysparycji (cięższe)

YOLO_EVERY_N = 5          # YOLO odpalaj co N klatek (np. 4-8)
YOLO_IMGSZ = 320          # mniejsze = szybciej (224/256/320)
YOLO_CONF = 0.5

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 9600

BASELINE_CM = 9.516
MODEL_PATH = "model_ai.onnx"

FRAME_W = 640
FRAME_H = 480

LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])

STOP_DIST_CM = 25.0

# Sterowanie (Twoje, przesunięte w prawo)
FORWARD_LEFT_CM  = 12
FORWARD_RIGHT_CM = 0
SHARP_LEFT_CM    = 20
SHARP_RIGHT_CM   = 14

# HSV “tracking” w ROI
HSV_MIN_RADIUS = 5
HSV_ERODE_IT = 1
HSV_DILATE_IT = 1


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

    return mapLx, mapLy, mapRx, mapRy, fx, cx


def send_uart(cmd_text, ser=None):
    if not (UART_ENABLED and ser is not None):
        return
    code = b"S"
    if cmd_text == "FORWARD": code = b"F"
    elif cmd_text == "LEFT": code = b"L"
    elif cmd_text == "SHARP LEFT": code = b"l"
    elif cmd_text == "RIGHT": code = b"R"
    elif cmd_text == "SHARP RIGHT": code = b"r"
    elif "STOP" in cmd_text: code = b"S"
    try:
        ser.write(code + b"\n")
    except Exception:
        pass


def detect_ai_left(model, frame):
    if model is None:
        return None, None
    # YOLO na pełnej klatce (dla max FPS możesz zrobić resize -> YOLO -> przeskalować bbox)
    results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            return (cx, cy), (x1, y1, x2, y2)
    return None, None


def hsv_center_in_roi(frame, roi):
    """Tani tracker HSV: zwraca (cx, cy) w układzie całej klatki."""
    x1, y1, x2, y2 = roi
    x1 = max(0, min(FRAME_W - 1, x1))
    x2 = max(0, min(FRAME_W, x2))
    y1 = max(0, min(FRAME_H - 1, y1))
    y2 = max(0, min(FRAME_H, y2))
    if x2 <= x1 or y2 <= y1:
        return None, None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    if HSV_ERODE_IT > 0:
        mask = cv2.erode(mask, None, iterations=HSV_ERODE_IT)
    if HSV_DILATE_IT > 0:
        mask = cv2.dilate(mask, None, iterations=HSV_DILATE_IT)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    c = max(contours, key=cv2.contourArea)
    (rx, ry), radius = cv2.minEnclosingCircle(c)
    if radius <= HSV_MIN_RADIUS:
        return None, None

    cx = int(rx) + x1
    cy = int(ry) + y1
    return (cx, cy), radius


def detect_hsv_right_from_left_bbox(frameR, bboxL):
    """Twoje główne podejście: ROI w prawym obrazie od x=0 do lx2+20 i y wokół bbox."""
    lx1, ly1, lx2, ly2 = bboxL
    y_start = max(0, ly1 - 40)
    y_end = min(FRAME_H, ly2 + 40)
    x_start = 0
    x_end = min(FRAME_W, lx2 + 20)
    roi = (x_start, y_start, x_end, y_end)
    center, _ = hsv_center_in_roi(frameR, roi)
    return center


def compute_X_center_cm(cx_L, cx_R, fx, cx0, baseline_cm):
    disp = float(cx_L - cx_R)
    if abs(disp) < 1e-6:
        return None, None, None
    Z = (fx * baseline_cm) / abs(disp)
    X_left = ((float(cx_L) - cx0) * Z) / fx
    X_center = X_left - (baseline_cm / 2.0)
    return X_center, Z, disp


def get_steering_command_from_X(X_center_cm):
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


def setup_camera(idx):
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    # zmniejsza opóźnienia bufora (nie zawsze działa na każdym backendzie)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # często pomaga na UVC:
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    return cap


def main():
    global FRAME_W, FRAME_H

    left_idx, right_idx, fw, fh = load_camera_config()
    FRAME_W, FRAME_H = fw, fh

    mapLx = mapLy = mapRx = mapRy = fx = cx0 = None
    if RECTIFY_ENABLED:
        mapLx, mapLy, mapRx, mapRy, fx, cx0 = load_stereo_calibration("stereoMap.xml")

    ser = None
    if UART_ENABLED:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.05)
            ser.flush()
        except Exception:
            ser = None

    model = None
    if os.path.isfile(MODEL_PATH):
        model = YOLO(MODEL_PATH, task="detect")

    capL = setup_camera(left_idx)
    capR = setup_camera(right_idx)

    if not capL.isOpened() or not capR.isOpened():
        print("FATAL: Cannot open cameras")
        sys.exit(1)

    # stan trackingu
    frame_id = 0
    bboxL = None          # ostatni bbox z YOLO
    last_centerL = None   # HSV tracking left
    last_cmd = "STOP"

    t_fps = time.perf_counter()
    fps = 0.0
    fps_cnt = 0

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not okL or not okR:
            break

        # Rektyfikacja tylko jeśli potrzebna (kosztowna)
        if RECTIFY_ENABLED:
            rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
            rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)
        else:
            rectL, rectR = frameL, frameR

        # 1) YOLO co N klatek lub gdy nie mamy bbox
        do_yolo = (bboxL is None) or (frame_id % YOLO_EVERY_N == 0)

        if do_yolo and model is not None:
            detL, new_bbox = detect_ai_left(model, rectL)
            if new_bbox is not None:
                bboxL = new_bbox
                last_centerL = detL
        else:
            # 2) HSV tracking na LEFT w małym ROI wokół poprzedniego bbox
            if bboxL is not None:
                x1, y1, x2, y2 = bboxL
                pad = 20
                roiL = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
                centerL, _ = hsv_center_in_roi(rectL, roiL)
                if centerL is not None:
                    last_centerL = centerL
                    # delikatnie “dociągnij” bbox do ROI (żeby ROI nie uciekało)
                    cx, cy = centerL
                    w = (x2 - x1)
                    h = (y2 - y1)
                    bboxL = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)

        cmd = "SEARCHING"
        dist_cm = 0.0

        # 3) Prawa kamera HSV na ROI wynikającym z bboxL
        if last_centerL is not None and bboxL is not None:
            detR = detect_hsv_right_from_left_bbox(rectR, bboxL)
            if detR is not None:
                cx_L, cy_L = last_centerL
                cx_R, cy_R = detR

                # Sterowanie po X_center: gdy nie ma rektyfikacji -> tylko “na oko”
                if RECTIFY_ENABLED and fx is not None and cx0 is not None:
                    Xc, Z, disp = compute_X_center_cm(cx_L, cx_R, fx, cx0, BASELINE_CM)
                    if Z is not None:
                        dist_cm = float(Z)
                    if Xc is not None:
                        cmd = get_steering_command_from_X(Xc)
                else:
                    # FAST: bez Z, tylko “kierunek” z różnicy pikseli (lepsze niż nic do debug)
                    # im większa różnica, tym mocniej skręca (tu tylko progi)
                    dx = cx_L - (FRAME_W // 2)
                    # mapowanie na cm “umowne” dla progów
                    Xc_fake = dx * 0.05
                    cmd = get_steering_command_from_X(Xc_fake)

                if RECTIFY_ENABLED and dist_cm > 0 and dist_cm < STOP_DIST_CM:
                    cmd = "STOP - OBSTACLE"
            else:
                cmd = "SEARCHING"
        else:
            cmd = "SEARCHING"

        # wysyłka UART tylko gdy zmiana (mniej spamowania)
        if cmd != last_cmd:
            send_uart(cmd, ser)
            last_cmd = cmd
            if TERMINAL_OUTPUT:
                print(f"KOMENDY: {last_cmd} -> {cmd}")
            last_cmd = cmd
        # ======= GUI =======
        if GUI_ENABLED:
            # overlay
            showL = rectL.copy()
            showR = rectR.copy()

            if bboxL is not None:
                x1, y1, x2, y2 = bboxL
                cv2.rectangle(showL, (max(0, x1), max(0, y1)), (min(FRAME_W - 1, x2), min(FRAME_H - 1, y2)), (0, 255, 0), 2)
            if last_centerL is not None:
                cv2.circle(showL, (int(last_centerL[0]), int(last_centerL[1])), 6, (0, 255, 255), 2)

            cv2.putText(showL, f"CMD: {cmd}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(showL, f"FPS: {fps:.1f}  YOLO every: {YOLO_EVERY_N}  RECT: {RECTIFY_ENABLED}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            if RECTIFY_ENABLED:
                cv2.putText(showL, f"Z: {dist_cm:.1f}cm", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            cv2.imshow("LEFT", showL)
            cv2.imshow("RIGHT", showR)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        # FPS licznik
        fps_cnt += 1
        if fps_cnt >= 10:
            now = time.perf_counter()
            dt = now - t_fps
            if dt > 0:
                fps = fps_cnt / dt
            fps_cnt = 0
            t_fps = now
            if TERMINAL_OUTPUT:
                print(f"CMD: {cmd:12s} | Z: {dist_cm:.1f} cm")
        frame_id += 1

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    if UART_ENABLED and ser is not None:
        ser.close()


if __name__ == "__main__":
    main()
