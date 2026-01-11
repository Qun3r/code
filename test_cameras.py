import cv2
import numpy as np

print("=" * 60)
print("TEST SUROWYCH OBRAZÓW Z KAMER (BEZ KALIBRACJI)")
print("=" * 60)

# Otwórz kamery
capL = cv2.VideoCapture(0)  # LEWA
capR = cv2.VideoCapture(2)  # PRAWA

capL.set(3, 640)
capL.set(4, 480)
capR.set(3, 640)
capR.set(4, 480)

if not capL.isOpened():
    print("❌ BŁĄD: Nie można otworzyć kamery index 0 (LEWA)")
else:
    print("✓ Kamera index 0 (LEWA) otwarta")

if not capR.isOpened():
    print("❌ BŁĄD: Nie można otworzyć kamery index 2 (PRAWA)")
else:
    print("✓ Kamera index 2 (PRAWA) otwarta")

print("\nNaciśnij 'q' aby zakończyć")
print("Naciśnij 's' aby zapisać obraz testowy")
print("=" * 60)

frame_count = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    
    if not retL:
        print("❌ Nie można odczytać z kamery LEWEJ!")
        break
    if not retR:
        print("❌ Nie można odczytać z kamery PRAWEJ!")
        break
    
    # Sprawdź czy obrazy nie są czarne
    meanL = frameL.mean()
    meanR = frameR.mean()
    
    # Dodaj informacje na obrazy
    cv2.putText(frameL, "LEWA (index 0) - RAW", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frameL, f"Brightness: {meanL:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(frameR, "PRAWA (index 2) - RAW", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frameR, f"Brightness: {meanR:.1f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Ostrzeżenia
    if meanL < 10:
        cv2.putText(frameL, "UWAGA: Obraz zbyt ciemny!", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if meanR < 10:
        cv2.putText(frameR, "UWAGA: Obraz zbyt ciemny!", (10, 450), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Wyświetl
    cv2.imshow("LEWA Kamera (RAW)", frameL)
    cv2.imshow("PRAWA Kamera (RAW)", frameR)
    
    # Co sekundę wypisz statystyki
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"LEWA: brightness={meanL:.1f} | PRAWA: brightness={meanR:.1f}")
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('test_left_raw.png', frameL)
        cv2.imwrite('test_right_raw.png', frameR)
        print("✓ Obrazy zapisane jako test_left_raw.png i test_right_raw.png")

capL.release()
capR.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("DIAGNOZA:")
print("=" * 60)
print("1. Czy widzisz NORMALNE obrazy z obu kamer?")
print("   - Jeśli TAK → Problem jest w kalibracji")
print("   - Jeśli NIE → Problem jest z kamerami/połączeniem")
print("")
print("2. Czy brightness jest >20?")
print("   - Jeśli NIE → Za ciemno, dodaj światło")
print("=" * 60)
