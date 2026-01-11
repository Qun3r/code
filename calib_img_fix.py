import cv2

# POPRAWKA: Teraz zgodne z vision.py
# capL (index 2) = LEWA kamera (patrząc od tyłu robota)
# capR (index 0) = PRAWA kamera (patrząc od tyłu robota)

capL = cv2.VideoCapture(2)  # LEWA
capR = cv2.VideoCapture(0)  # PRAWA

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

num = 0

print("=" * 60)
print("KALIBRACJA - ZAPIS OBRAZÓW")
print("=" * 60)
print("LEWA kamera (index 2) = lewa strona robota (patrząc od tyłu)")
print("PRAWA kamera (index 0) = prawa strona robota (patrząc od tyłu)")
print("=" * 60)
print("Naciśnij 's' aby zapisać parę zdjęć")
print("Naciśnij ESC aby zakończyć")
print("=" * 60)

while capL.isOpened():

    succes1, imgL = capL.read()
    succes2, imgR = capR.read()

    if not succes1 or not succes2:
        print("BŁĄD: Nie można odczytać z kamer!")
        break

    # Dodaj etykiety na obrazach
    imgL_display = imgL.copy()
    imgR_display = imgR.copy()
    cv2.putText(imgL_display, "LEWA (index 2)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(imgR_display, "PRAWA (index 0)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(imgL_display, f"Zapisane pary: {num}", (10, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    k = cv2.waitKey(5)

    if k == 27:  # ESC
        break
    elif k == ord('s'):  # Zapisz
        cv2.imwrite('Calibration/images/stereoLeft/imageL' + str(num) + '.png', imgL)
        cv2.imwrite('Calibration/images/stereoRight/imageR' + str(num) + '.png', imgR)
        print(f"✓ Para {num} zapisana! (LEWA=index2, PRAWA=index0)")
        num += 1

    cv2.imshow('LEWA Kamera (index 2)', imgL_display)
    cv2.imshow('PRAWA Kamera (index 0)', imgR_display)

print(f"\nZapisano łącznie {num} par zdjęć")
print("UWAGA: Teraz uruchom stereovision_calibration.py")

capL.release()
capR.release()
cv2.destroyAllWindows()
