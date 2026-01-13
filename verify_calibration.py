import cv2
import json
import numpy as np
import os

CFG = json.load(open("camera_config.json", "r", encoding="utf-8"))
L_IDX = CFG["left_index"]
R_IDX = CFG["right_index"]
W = CFG.get("frame_w", 640)
H = CFG.get("frame_h", 480)

XML = "stereoMap.xml"
if not os.path.isfile(XML):
    print("ERROR: brak stereoMap.xml, uruchom stereo_calibrate.py")
    raise SystemExit

fs = cv2.FileStorage(XML, cv2.FILE_STORAGE_READ)
mapLx = fs.getNode("stereoMapL_x").mat()
mapLy = fs.getNode("stereoMapL_y").mat()
mapRx = fs.getNode("stereoMapR_x").mat()
mapRy = fs.getNode("stereoMapR_y").mat()
fs.release()

capL = cv2.VideoCapture(L_IDX)
capR = cv2.VideoCapture(R_IDX)
capL.set(cv2.CAP_PROP_FRAME_WIDTH, W); capL.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, W); capR.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not capL.isOpened() or not capR.isOpened():
    print("ERROR: nie mogę otworzyć kamer.")
    raise SystemExit

print("q = wyjście")

def draw_epilines(img, step=40):
    out = img.copy()
    for y in range(0, H, step):
        cv2.line(out, (0, y), (W, y), (0, 255, 255), 1)
    cv2.line(out, (W//2, 0), (W//2, H), (255, 0, 255), 1)
    return out

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    rectL = draw_epilines(rectL, 40)
    rectR = draw_epilines(rectR, 40)

    cv2.putText(rectL, f"RECT LEFT index={L_IDX}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(rectR, f"RECT RIGHT index={R_IDX}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Rectified LEFT", rectL)
    cv2.imshow("Rectified RIGHT", rectR)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
