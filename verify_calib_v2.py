import cv2
import numpy as np
import sys
import os

print("=" * 70)
print("WERYFIKACJA KALIBRACJI STEREO")
print("=" * 70)

# 1. Sprawdź czy plik istnieje
if not os.path.isfile('stereoMap.xml'):
    print("❌ BŁĄD: Nie znaleziono pliku stereoMap.xml!")
    print("   Najpierw uruchom stereovision_calibration.py")
    sys.exit()

# 2. Wczytaj mapy kalibracyjne
print("\n[1/5] Wczytywanie map kalibracyjnych...")
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_READ)

mapLx = cv_file.getNode('stereoMapL_x').mat()
mapLy = cv_file.getNode('stereoMapL_y').mat()
mapRx = cv_file.getNode('stereoMapR_x').mat()
mapRy = cv_file.getNode('stereoMapR_y').mat()
cv_file.release()

if mapLx is None or mapRx is None:
    print("❌ BŁĄD: Nie można wczytać map kalibracyjnych!")
    sys.exit()

print("✓ Mapy wczytane pomyślnie")
print(f"   Rozmiar mapy lewej: {mapLx.shape}")
print(f"   Rozmiar mapy prawej: {mapRx.shape}")

# 3. Otwórz kamery (POPRAWIONA KOLEJNOŚĆ)
print("\n[2/5] Otwieranie kamer...")
capL = cv2.VideoCapture(0)  # LEWA
capR = cv2.VideoCapture(2)  # PRAWA

capL.set(3, 640)
capL.set(4, 480)
capR.set(3, 640)
capR.set(4, 480)

if not capL.isOpened() or not capR.isOpened():
    print("❌ BŁĄD: Nie można otworzyć kamer!")
    sys.exit()

print("✓ Kamery otwarte (LEWA=index0, PRAWA=index2)")

# 4. Test rektyfikacji
print("\n[3/5] Testowanie rektyfikacji...")

retL, frameL = capL.read()
retR, frameR = capR.read()

if not retL or not retR:
    print("❌ BŁĄD: Nie można odczytać obrazów z kamer!")
    sys.exit()

# Rektyfikuj
rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

print("✓ Rektyfikacja działa")

# 5. Analiza jakości
print("\n[4/5] Analiza jakości kalibracji...")

# Sprawdź czy obrazy są podobne (nie są czarne/białe)
if rectL.mean() < 10 or rectL.mean() > 245:
    print("⚠ OSTRZEŻENIE: Obraz lewy może być uszkodzony (zbyt ciemny/jasny)")
if rectR.mean() < 10 or rectR.mean() > 245:
    print("⚠ OSTRZEŻENIE: Obraz prawy może być uszkodzony (zbyt ciemny/jasny)")

# Sprawdź rozmiary
if rectL.shape != rectR.shape:
    print("❌ BŁĄD: Obrazy mają różne rozmiary po rektyfikacji!")
else:
    print(f"✓ Rozmiary zgodne: {rectL.shape}")

# 6. Wizualizacja z liniami epipolarnymi
print("\n[5/5] Wizualizacja...")

# Stwórz obraz z liniami poziomymi co 30px
combined = np.hstack((rectL, rectR))
for i in range(0, 480, 30):
    cv2.line(combined, (0, i), (1280, i), (0, 255, 0), 1)

# Dodaj etykiety
cv2.putText(combined, "LEWA (index 0)", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(combined, "PRAWA (index 2)", (650, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(combined, "Zielone linie powinny byc ROWNOLEGLE", (200, 460), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

print("\n" + "=" * 70)
print("WYNIKI:")
print("=" * 70)
print("✓ Kalibracja załadowana poprawnie")
print("✓ Kamery działają")
print("✓ Rektyfikacja działa")
print("\nSPRAWDŹ W OKNIE:")
print("1. Czy te same obiekty są na tej samej wysokości w obu obrazach?")
print("2. Czy zielone linie przechodzą przez te same punkty?")
print("3. Jeśli NIE - kalibracja jest ZŁA, powtórz proces!")
print("\nNaciśnij 'q' aby zakończyć, 's' aby zapisać obraz testowy")
print("=" * 70)

# Pokaż wynik
cv2.imshow("WERYFIKACJA KALIBRACJI - Linie Epipolaryne", combined)

while True:
    # Aktualizuj obraz w pętli
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    
    if retL and retR:
        rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)
        
        combined = np.hstack((rectL, rectR))
        for i in range(0, 480, 30):
            cv2.line(combined, (0, i), (1280, i), (0, 255, 0), 1)
        
        cv2.putText(combined, "LEWA (index 0)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(combined, "PRAWA (index 2)", (650, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("WERYFIKACJA KALIBRACJI - Linie Epipolaryne", combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('calibration_verification.png', combined)
        print("✓ Obraz zapisany jako calibration_verification.png")

capL.release()
capR.release()
cv2.destroyAllWindows()

print("\n✓ Test zakończony")
