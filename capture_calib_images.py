import cv2
import json
import os
import time

CFG = json.load(open("camera_config.json", "r", encoding="utf-8"))
L_IDX = CFG["left_index"]
R_IDX = CFG["right_index"]
W = CFG.get("frame_w", 640)
H = CFG.get("frame_h", 480)

OUT_DIR = "calib_images"
os.makedirs(OUT_DIR, exist_ok=True)

CHESSBOARD_SIZE = (8, 5)    # <-- DOPASUJ do swojej szachownicy (wewn. narożniki)
SHOW_CORNERS = True

capL = cv2.VideoCapture(L_IDX)
capR = cv2.VideoCapture(R_IDX)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, W); capL.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, W); capR.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not capL.isOpened() or not capR.isOpened():
    print("ERROR: nie mogę otworzyć kamer.")
    raise SystemExit

count = 0
print("s = zapisz parę, q = wyjście")
print("Stań OD TYŁU robota: LEFT to lewa strona robota, RIGHT to prawa.")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("ERROR: odczyt z kamer nie działa.")
        break

    dispL = frameL.copy()
    dispR = frameR.copy()

    if SHOW_CORNERS:
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
        okL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
        okR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)
        if okL:
            cv2.drawChessboardCorners(dispL, CHESSBOARD_SIZE, cornersL, okL)
        if okR:
            cv2.drawChessboardCorners(dispR, CHESSBOARD_SIZE, cornersR, okR)

    cv2.putText(dispL, f"LEFT index={L_IDX}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(dispR, f"RIGHT index={R_IDX}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("LEFT", dispL)
    cv2.imshow("RIGHT", dispR)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('s'):
        ts = int(time.time() * 1000)
        left_path = os.path.join(OUT_DIR, f"left_{count:04d}_{ts}.png")
        right_path = os.path.join(OUT_DIR, f"right_{count:04d}_{ts}.png")
        cv2.imwrite(left_path, frameL)
        cv2.imwrite(right_path, frameR)
        print("Zapisano:", left_path, right_path)
        count += 1

capL.release()
capR.release()
cv2.destroyAllWindows()
