import numpy as np
import cv2 as cv
import glob
import sys

print("=" * 70)
print("ULEPSZONA KALIBRACJA STEREO Z DIAGNOSTYKĄ")
print("=" * 70)

################ KONFIGURACJA #############################

################ DETEKCJA ROZMIARU SZACHOWNICY #############################

# Możliwe rozmiary dla szachownicy 6x9 kwadratów (czyli 5x8 narożników)
possible_sizes = [
    (8, 5),  # 9x6 kwadratów poziomo
    (5, 8),  # 6x9 kwadratów pionowo
]

frameSize = (640, 480)
size_of_chessboard_squares_mm = 30  # [mm] - ZMIEŃ jeśli inny!

print(f"\nKONFIGURACJA:")
print(f"  Szachownica: 6x9 kwadratów = 5x8 narożników wewnętrznych")
print(f"  Rozmiar kwadratu: {size_of_chessboard_squares_mm}mm")
print(f"  Rozmiar ramki: {frameSize}")
print(f"  Próbuję wykryć orientację szachownicy...")
print("=" * 70)

################ PRZYGOTOWANIE PUNKTÓW #############################

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Tablice do przechowywania punktów
objpoints = []     # 3D punkty w przestrzeni rzeczywistej
imgpointsL = []    # 2D punkty na obrazie LEWYM
imgpointsR = []    # 2D punkty na obrazie PRAWYM

# AUTO-DETEKCJA rozmiaru szachownicy
chessboardSize = None

################ WCZYTANIE ZDJĘĆ #############################

imagesLeft = sorted(glob.glob('Calibration/images/stereoLeft/*.png'))
imagesRight = sorted(glob.glob('Calibration/images/stereoRight/*.png'))

if len(imagesLeft) == 0 or len(imagesRight) == 0:
    print("❌ BŁĄD: Brak zdjęć kalibracyjnych!")
    print("   Uruchom najpierw: calibration_images_FIXED.py")
    sys.exit()

if len(imagesLeft) != len(imagesRight):
    print(f"❌ BŁĄD: Różna liczba zdjęć!")
    print(f"   Lewe: {len(imagesLeft)}, Prawe: {len(imagesRight)}")
    sys.exit()

print(f"\n[1/6] Znaleziono {len(imagesLeft)} par zdjęć")

if len(imagesLeft) < 10:
    print("⚠ OSTRZEŻENIE: Za mało zdjęć! Minimum 15 par zalecane.")

################ ZNAJDOWANIE NAROŻNIKÓW SZACHOWNICY #############################

print("\n[2/6] Szukanie narożników szachownicy...")

successful_pairs = 0
failed_pairs = []

for idx, (imgLeft, imgRight) in enumerate(zip(imagesLeft, imagesRight)):
    
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    
    if imgL is None or imgR is None:
        print(f"  ❌ Para {idx}: Nie można wczytać obrazów")
        failed_pairs.append(idx)
        continue
    
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # AUTO-DETEKCJA rozmiaru - spróbuj obu orientacji
    if chessboardSize is None:
        for test_size in possible_sizes:
            retL_test, _ = cv.findChessboardCorners(grayL, test_size, None)
            retR_test, _ = cv.findChessboardCorners(grayR, test_size, None)
            
            if retL_test and retR_test:
                chessboardSize = test_size
                print(f"  ✓ Wykryto rozmiar szachownicy: {chessboardSize[0]}x{chessboardSize[1]} narożników")
                
                # Przygotuj punkty 3D
                objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
                objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
                objp = objp * size_of_chessboard_squares_mm
                break
        
        if chessboardSize is None:
            print(f"  ❌ Para {idx}: Nie można wykryć rozmiaru szachownicy!")
            print("     Sprawdź czy szachownica ma 6x9 kwadratów (5x8 narożników)")
            failed_pairs.append(idx)
            continue

    # Znajdź narożniki
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    if retL and retR:
        # Obie kamery znalazły szachownicę
        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        successful_pairs += 1
        print(f"  ✓ Para {idx}: Szachownica znaleziona w OBUDU kamerach")
        
    else:
        print(f"  ❌ Para {idx}: Szachownica NIE znaleziona (L={retL}, R={retR})")
        failed_pairs.append(idx)

cv.destroyAllWindows()

print(f"\nWYNIK DETEKCJI:")
print(f"  Wykryty rozmiar: {chessboardSize[0]}x{chessboardSize[1]} narożników")
print(f"  ✓ Udane pary: {successful_pairs}")
print(f"  ❌ Nieudane pary: {len(failed_pairs)}")

if chessboardSize is None:
    print("\n❌ KRYTYCZNY BŁĄD: Nie można wykryć szachownicy!")
    print("   SPRAWDŹ:")
    print("   1. Czy szachownica to 6x9 kwadratów (5x8 narożników)?")
    print("   2. Czy jest dobrze oświetlona?")
    print("   3. Czy jest widoczna w OBUDU kamerach?")
    sys.exit()

if successful_pairs < 10:
    print("\n❌ KRYTYCZNY BŁĄD: Za mało udanych par (<10)!")
    print("   ROZWIĄZANIE:")
    print("   1. Zrób WIĘCEJ zdjęć z calibration_images_FIXED.py")
    print("   2. Upewnij się że szachownica jest DOBRZE OŚWIETLONA")
    print("   3. Szachownica musi być widoczna w OBUDU kamerach jednocześnie")
    sys.exit()

################ KALIBRACJA POJEDYNCZYCH KAMER #############################

print("\n[3/6] Kalibracja lewej kamery...")
retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
    objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(
    cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

print(f"  ✓ RMS błąd: {retL:.4f}")

print("\n[4/6] Kalibracja prawej kamery...")
retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
    objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(
    cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

print(f"  ✓ RMS błąd: {retR:.4f}")

################ KALIBRACJA STEREO #############################

print("\n[5/6] Kalibracja stereo (powiązanie kamer)...")

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, 
    newCameraMatrixL, distL, 
    newCameraMatrixR, distR, 
    grayL.shape[::-1], 
    criteria_stereo, 
    flags)

print(f"  ✓ RMS błąd stereo: {retStereo:.4f}")

if retStereo > 1.0:
    print("  ⚠ OSTRZEŻENIE: Wysoki błąd RMS (>1.0)!")
    print("    Kalibracja może być niedokładna. Rozważ ponowne zdjęcia.")

################ REKTYFIKACJA STEREO #############################

print("\n[6/6] Rektyfikacja stereo...")

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
    newCameraMatrixL, distL, 
    newCameraMatrixR, distR, 
    grayL.shape[::-1], 
    rot, trans, 
    rectifyScale, (0,0))

stereoMapL = cv.initUndistortRectifyMap(
    newCameraMatrixL, distL, rectL, projMatrixL, 
    grayL.shape[::-1], cv.CV_16SC2)

stereoMapR = cv.initUndistortRectifyMap(
    newCameraMatrixR, distR, rectR, projMatrixR, 
    grayR.shape[::-1], cv.CV_16SC2)

################ ZAPIS DO PLIKU #############################

print("\nZapis parametrów do stereoMap.xml...")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x', stereoMapL[0])
cv_file.write('stereoMapL_y', stereoMapL[1])
cv_file.write('stereoMapR_x', stereoMapR[0])
cv_file.write('stereoMapR_y', stereoMapR[1])

cv_file.release()

print("✓ Zapisano!")

################ PODSUMOWANIE #############################

print("\n" + "=" * 70)
print("PODSUMOWANIE KALIBRACJI")
print("=" * 70)
print(f"Udanych par zdjęć: {successful_pairs}")
print(f"Błąd RMS lewej kamery: {retL:.4f}")
print(f"Błąd RMS prawej kamery: {retR:.4f}")
print(f"Błąd RMS stereo: {retStereo:.4f}")
print("")
print("Baseline (odległość między kamerami):")
baseline_cm = np.linalg.norm(trans) / 10  # z mm na cm
print(f"  {baseline_cm:.2f} cm")
print("")

if retStereo < 0.5:
    print("✓✓✓ DOSKONAŁA kalibracja! (RMS < 0.5)")
elif retStereo < 1.0:
    print("✓✓ DOBRA kalibracja (RMS < 1.0)")
else:
    print("⚠ PRZECIĘTNA kalibracja (RMS > 1.0)")
    print("  Rozważ ponowne zdjęcia dla lepszej jakości")

print("\nNastępny krok:")
print("  python verify_calibration.py")
print("=" * 70)
