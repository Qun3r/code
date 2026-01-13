import cv2
import numpy as np
import glob
import os
import json

CFG = json.load(open("camera_config.json", "r", encoding="utf-8"))
W = CFG.get("frame_w", 640)
H = CFG.get("frame_h", 480)

CHESSBOARD_SIZE = (9, 6)     # <-- DOPASUJ (wewn. narożniki)
SQUARE_SIZE_CM = 2.5         # <-- DOPASUJ (w cm)
BASELINE_CM = 9.5            # u Ciebie

IMG_DIR = "calib_images"
OUT_XML = "stereoMap.xml"

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)

objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_CM

left_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "left_*.png")))
right_imgs = sorted(glob.glob(os.path.join(IMG_DIR, "right_*.png")))

if len(left_imgs) == 0 or len(right_imgs) == 0:
    print("ERROR: brak zdjęć w calib_images.")
    raise SystemExit

pairs = min(len(left_imgs), len(right_imgs))
left_imgs = left_imgs[:pairs]
right_imgs = right_imgs[:pairs]

objpoints = []
imgpointsL = []
imgpointsR = []

used = 0
for lp, rp in zip(left_imgs, right_imgs):
    imgL = cv2.imread(lp)
    imgR = cv2.imread(rp)
    if imgL is None or imgR is None:
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    okL, cornersL = cv2.findChessboardCorners(grayL, CHESSBOARD_SIZE, None)
    okR, cornersR = cv2.findChessboardCorners(grayR, CHESSBOARD_SIZE, None)
    if not okL or not okR:
        continue

    cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

    objpoints.append(objp)
    imgpointsL.append(cornersL)
    imgpointsR.append(cornersR)
    used += 1

print("Użyte pary do kalibracji:", used, "z", pairs)
if used < 10:
    print("ERROR: za mało dobrych par (min 10, lepiej 20-30).")
    raise SystemExit

# 1) Kalibracja każdej kamery osobno
retL, cameraMatrixL, distCoeffsL, rvecsL, tvecsL = cv2.calibrateCamera(
    objpoints, imgpointsL, (W, H), None, None
)
retR, cameraMatrixR, distCoeffsR, rvecsR, tvecsR = cv2.calibrateCamera(
    objpoints, imgpointsR, (W, H), None, None
)

# 2) StereoCalibrate
flags = (cv2.CALIB_FIX_INTRINSIC)
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7)

retS, CM1, DC1, CM2, DC2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    cameraMatrixL, distCoeffsL,
    cameraMatrixR, distCoeffsR,
    (W, H),
    criteria=stereo_criteria,
    flags=flags
)

# 3) StereoRectify
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    CM1, DC1, CM2, DC2, (W, H), R, T, alpha=0
)

# 4) Mapy rektyfikacji
stereoMapL = cv2.initUndistortRectifyMap(CM1, DC1, R1, P1, (W, H), cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(CM2, DC2, R2, P2, (W, H), cv2.CV_16SC2)

# 5) Zapis pełny do XML
fs = cv2.FileStorage(OUT_XML, cv2.FILE_STORAGE_WRITE)

fs.write("stereoMapL_x", stereoMapL[0])
fs.write("stereoMapL_y", stereoMapL[1])
fs.write("stereoMapR_x", stereoMapR[0])
fs.write("stereoMapR_y", stereoMapR[1])

fs.write("cameraMatrixL", CM1)
fs.write("distCoeffsL", DC1)
fs.write("cameraMatrixR", CM2)
fs.write("distCoeffsR", DC2)

fs.write("R", R)
fs.write("T", T)
fs.write("E", E)
fs.write("F", F)

fs.write("R1", R1)
fs.write("R2", R2)
fs.write("projMatrixL", P1)
fs.write("projMatrixR", P2)
fs.write("Q", Q)

fs.release()

fx = float(P1[0, 0])
cx = float(P1[0, 2])
print("✓ Zapisano", OUT_XML)
print(f"Z kalibracji: fx={fx:.2f}px, cx={cx:.2f}px")
print(f"T (cm): [{T[0,0]:.3f}, {T[1,0]:.3f}, {T[2,0]:.3f}]  | baseline ~ {abs(T[0,0]):.3f} cm (powinno być ~{BASELINE_CM})")
