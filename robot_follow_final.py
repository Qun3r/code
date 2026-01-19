import cv2
import numpy as np
import sys
import os
import time
import json
from ultralytics import YOLO
import serial


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

CENTER_MARGIN_CM = 6
SHARP_TURN_CM = 14
STOP_DIST_CM = 25.0
#dodane
FORWARD_LEFT_CM  = 10  # ile cm w lewo nadal uznajemy za FORWARD (większe = mniej LEFT)
FORWARD_RIGHT_CM = 3  # ile cm w prawo nadal FORWARD (większe = przesunięcie strefy w prawo)

SHARP_LEFT_CM  = 18
SHARP_RIGHT_CM = 12  

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
    # strefa FORWARD jest szersza i przesunięta w prawo:
    # FORWARD gdy X jest w przedziale [-FORWARD_LEFT_CM, +FORWARD_RIGHT_CM]
    if -FORWARD_LEFT_CM <= X_center_cm <= FORWARD_RIGHT_CM:
        return "FORWARD"

    # LEFT strona (łatwiej łapie LEFT jeśli wyjdziesz poza lewy próg)
    if X_center_cm < -SHARP_LEFT_CM:
        return "SHARP LEFT"
    if X_center_cm < -FORWARD_LEFT_CM:
        return "LEFT"

    # RIGHT strona (łatwiej skręcić w prawo jeśli dasz mniejsze progi)
    if X_center_cm > SHARP_RIGHT_CM:
        return "SHARP RIGHT"
    if X_center_cm > FORWARD_RIGHT_CM:
        return "RIGHT"

    return "FORWARD"

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
        print(f"BASELINE={BASELINE_CM} cm | baseline/2={BASELINE_CM/2:.2f} cm")
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

    if TERMINAL_OUTPUT:
        print("SYSTEM STARTED - press 'q' to quit")

    while True:
        t0 = time.time()

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
        cx_L = None
        cx_R = None

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

            if TERMINAL_OUTPUT and cx_R is not None and X_center is not None:
                print(f"DBG cxL={cx_L} cxR={cx_R} disp={disp:.1f}  Xc={X_center:.2f}cm  Z={dist_cm:.1f}cm")

            process_output(cmd, dist_cm, ser)
        else:
            process_output("SEARCHING", 0.0, ser)
            
        if GUI_ENABLED:
            fps = 1.0 / max(1e-6, (time.time() - t0))

            # strefy w cm (wizualizacja tylko orientacyjna) -> rysujemy linie w pikselach wokół osi cx0
            # Wyznacz szerokość marginesu w px przy założeniu dystansu ~100cm (tylko do podglądu)
            approx_Z = 100.0
            px_per_cm = fx / approx_Z
            center_px = int(cx0)
            margin_px = int(CENTER_MARGIN_CM * px_per_cm)
            sharp_px = int(SHARP_TURN_CM * px_per_cm)

            cv2.line(rectL, (center_px, 0), (center_px, FRAME_H), (255, 0, 255), 1)
            cv2.line(rectL, (center_px - margin_px, 0), (center_px - margin_px, FRAME_H), (0, 0, 255), 1)
            cv2.line(rectL, (center_px + margin_px, 0), (center_px + margin_px, FRAME_H), (0, 0, 255), 1)
            cv2.line(rectL, (center_px - sharp_px, 0), (center_px - sharp_px, FRAME_H), (0, 0, 255), 2)
            cv2.line(rectL, (center_px + sharp_px, 0), (center_px + sharp_px, FRAME_H), (0, 0, 255), 2)

            cv2.putText(rectL, f"LEFT idx={left_idx} (AI)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(rectR, f"RIGHT idx={right_idx} (HSV)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.putText(rectL, f"FPS:{fps:.1f} fx:{fx:.0f}px cx:{cx0:.0f}px",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            if detL is not None and bboxL is not None:
                x1, y1, x2, y2 = bboxL
                cv2.rectangle(rectL, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rectL, cmd, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if cx_L is not None:
                cv2.circle(rectL, (int(cx_L), int(cy_L)), 6, (0, 255, 255), 2)

            if cx_R is not None:
                cv2.circle(rectR, (int(cx_R), int(cy_R)), 6, (0, 255, 255), 2)

            if X_center is not None:
                cv2.putText(rectL, f"X_center={X_center:.2f}cm  Z={dist_cm:.1f}cm  disp={disp:.1f}px",
                            (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Left (rectified)", rectL)
            cv2.imshow("Right (rectified)", rectR)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    if UART_ENABLED and ser is not None:
        ser.close()


if __name__ == "__main__":
    main()
