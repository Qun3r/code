import cv2
import numpy as np
import sys
import os
import time
import json
from ultralytics import YOLO


# =========================
# TRYB TESTU C (ODLEGLOSC)
# =========================
TEST_C_ENABLED = True

# docelowa liczba POPRAWNYCH pomiarów (detL+detR+dysparycja)
TEST_C_TARGET_VALID = 100

# co ile poprawnych pomiarów wypisać statystyki
TEST_C_PRINT_EVERY_VALID = 100

# zabezpieczenie: maksymalna liczba iteracji pętli (gdyby detekcje "nie łapały")
TEST_C_HARD_MAX_ITERS = 1000

# odległość referencyjna w cm (możesz też podać jako argv[1])
Z_REF_CM_DEFAULT = 200

# zapis do CSV (polecam do tabel w rozdz. 7)
CSV_LOG_ENABLED = True
CSV_PATH = "testC_distance_log.csv"


GUI_ENABLED = False
TERMINAL_OUTPUT = True

BASELINE_CM = 9.5
MODEL_PATH = "model_ai.onnx"

FRAME_W = 640
FRAME_H = 480

LOWER_HSV = np.array([29, 86, 6])
UPPER_HSV = np.array([64, 255, 255])


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


def detect_ai_left(model, frame):
    """
    Zwraca:
      - det: (cx, cy)
      - bbox: (x1, y1, x2, y2)
      - conf: float
    """
    if model is None:
        return None, None, None

    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    best = None  # (conf, det, bbox)
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            candidate = (conf, (cx, cy), (x1, y1, x2, y2))
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None, None, None

    conf, det, bbox = best
    return det, bbox, conf


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

def compute_depth_cm(cx_L, cx_R, fx, baseline_cm):
    """
    Zwraca: Z [cm], disp [px]
    """
    disp = float(cx_L - cx_R)
    if abs(disp) < 1e-6:
        return None, None

    Z = (fx * baseline_cm) / abs(disp)
    return float(Z), float(disp)


def stats_from_samples(z_list_cm, z_ref_cm):
    """
    Liczy:
      meanZ, stdZ, mean_abs_error, rel_error_percent
    """
    arr = np.array(z_list_cm, dtype=np.float64)
    meanZ = float(arr.mean())
    stdZ = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    err = np.abs(arr - float(z_ref_cm))
    mean_abs_error = float(err.mean())
    rel_error_percent = float((mean_abs_error / float(z_ref_cm)) * 100.0) if z_ref_cm > 0 else 0.0

    return meanZ, stdZ, mean_abs_error, rel_error_percent


def print_testC_stats(valid_count, total_iters, z_list_cm, z_ref_cm, prefix="TEST C"):
    meanZ, stdZ, mean_abs_error, rel_error_percent = stats_from_samples(z_list_cm, z_ref_cm)
    print(
        f"[{prefix}] valid={valid_count}/{total_iters} | "
        f"Zref={z_ref_cm:.1f}cm | meanZ={meanZ:.1f}cm | stdZ={stdZ:.2f}cm | "
        f"mean|e|={mean_abs_error:.2f}cm | e%={rel_error_percent:.2f}%"
    )


def main():
    # ====== Zref: z argumentu lub domyślnie ======
    z_ref_cm = Z_REF_CM_DEFAULT
    if len(sys.argv) >= 2:
        try:
            z_ref_cm = float(sys.argv[1])
        except ValueError:
            print("Użycie: python3 robot_stereo_follow_testC_distance.py <Z_REF_CM>")
            sys.exit(1)

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
        print("=" * 70)
        print("CONFIG (TEST C - DISTANCE):")
        print(f"Z_REF = {z_ref_cm:.1f} cm")
        print(f"LEFT camera index : {left_idx}")
        print(f"RIGHT camera index: {right_idx}")
        print(f"Frame: {FRAME_W}x{FRAME_H}")
        print(f"fx={fx:.2f}px, cx0={cx0:.2f}px (from stereoMap.xml)")
        print(f"BASELINE={BASELINE_CM} cm")
        print(f"TARGET_VALID={TEST_C_TARGET_VALID} | print every valid={TEST_C_PRINT_EVERY_VALID} | hard max iters={TEST_C_HARD_MAX_ITERS}")
        print("=" * 70)

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
        sys.exit(1)

    capL = cv2.VideoCapture(left_idx)
    capR = cv2.VideoCapture(right_idx)
    capL.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capL.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    capR.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    capR.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not capL.isOpened() or not capR.isOpened():
        print("FATAL ERROR: Cannot open cameras!")
        sys.exit(1)

    # ====== log CSV ======
    fcsv = None
    if CSV_LOG_ENABLED:
        fcsv = open(CSV_PATH, "w", encoding="utf-8")
        fcsv.write("t_ms,valid,Z_cm,disp_px,cxL,cxR,conf\n")

    # ====== liczniki i próbki ======
    total_iters = 0
    valid = 0
    z_samples = []  # lista Z_cm dla poprawnych detekcji

    if TERMINAL_OUTPUT:
        print("TEST C STARTED - ustaw pilke na Z_REF i poczekaj na zebrane probki...")

    while True:
        total_iters += 1
        if total_iters > TEST_C_HARD_MAX_ITERS:
            print("[TEST C] HARD STOP: za malo poprawnych pomiarow (sprawdz detekcje / oswietlenie).")
            break

        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            print("CRITICAL: Camera read failed!")
            break

        rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

        detL, bboxL, conf = detect_ai_left(model, rectL)
        detR = None
        Z_cm = None
        disp = None

        ok = False
        if detL is not None and bboxL is not None:
            cxL, cyL = detL
            detR = detect_hsv_smart_right(rectR, bboxL, FRAME_W, FRAME_H)
            if detR is not None:
                cxR, cyR = detR
                Z_cm, disp = compute_depth_cm(cxL, cxR, fx, BASELINE_CM)
                if Z_cm is not None:
                    ok = True

        if ok:
            valid += 1
            z_samples.append(Z_cm)

        # CSV log
        if fcsv is not None:
            t_ms = int(time.time() * 1000)
            if detL is not None:
                cxL = detL[0]
            else:
                cxL = ""
            if detR is not None:
                cxR = detR[0]
            else:
                cxR = ""
            conf_out = f"{conf:.3f}" if conf is not None else ""
            z_out = f"{Z_cm:.3f}" if Z_cm is not None else ""
            disp_out = f"{disp:.3f}" if disp is not None else ""
            fcsv.write(f"{t_ms},{1 if ok else 0},{z_out},{disp_out},{cxL},{cxR},{conf_out}\n")

        # wypis statystyk co N poprawnych pomiarów
        if valid > 0 and (valid % TEST_C_PRINT_EVERY_VALID) == 0:
            print_testC_stats(valid, total_iters, z_samples, z_ref_cm, prefix="TEST C (DISTANCE)")

        # stop po zebraniu docelowej liczby poprawnych pomiarów
        if valid >= TEST_C_TARGET_VALID:
            print("[TEST C] Done.")
            break

    # finalne statystyki (na koniec zawsze)
    if valid > 0:
        print_testC_stats(valid, total_iters, z_samples, z_ref_cm, prefix="TEST C FINAL")

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    if fcsv is not None:
        fcsv.close()
        print(f"[TEST C] CSV saved: {CSV_PATH}")


if __name__ == "__main__":
    main()
