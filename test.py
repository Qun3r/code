import cv2
import numpy as np

print("=" * 70)
print("DIAGNOZA UKŁADU KAMER")
print("=" * 70)
print("\nTen skrypt pomoże ustalić:")
print("1. Która kamera jest gdzie FIZYCZNIE")
print("2. Jak powinniśmy liczyć offset")
print("\n" + "=" * 70)

# Otwórz kamery
capL = cv2.VideoCapture(0)  # Nazywamy "Lewa"
capR = cv2.VideoCapture(2)  # Nazywamy "Prawa"

capL.set(3, 640)
capL.set(4, 480)
capR.set(3, 640)
capR.set(4, 480)

if not capL.isOpened() or not capR.isOpened():
    print("BŁĄD: Nie można otworzyć kamer!")
    exit()

print("\nINSTRUKCJE:")
print("1. Postaw piłkę DOKŁADNIE NA ŚRODKU robota (patrząc od tyłu)")
print("2. Piłka powinna być ~50-100cm przed robotem")
print("3. Obserwuj GDZIE piłka się pojawia w każdym oknie")
print("4. Zapisz pozycje X (cyferki na górze okien)")
print("\nNaciśnij 's' aby zapisać snapshot, 'q' aby zakończyć")
print("=" * 70)

frame_count = 0

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    
    if not retL or not retR:
        print("Błąd odczytu!")
        break
    
    # Dodaj linie referencyjne i info
    h, w = frameL.shape[:2]
    
    # Kamera "Lewa" (index 0)
    cv2.line(frameL, (w//2, 0), (w//2, h), (0, 255, 0), 2)  # Zielona linia środka
    cv2.circle(frameL, (w//2, h//2), 10, (0, 255, 0), 2)
    
    cv2.putText(frameL, "KAMERA INDEX 0", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frameL, "SRODEK = X:320", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Kamera "Prawa" (index 2)
    cv2.line(frameR, (w//2, 0), (w//2, h), (0, 0, 255), 2)  # Czerwona linia środka
    cv2.circle(frameR, (w//2, h//2), 10, (0, 0, 255), 2)
    
    cv2.putText(frameR, "KAMERA INDEX 2", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frameR, "SRODEK = X:320", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Pokazuj pozycje myszy (dla debugowania)
    frame_count += 1
    
    cv2.imshow("INDEX 0 (lewa w kodzie)", frameL)
    cv2.imshow("INDEX 2 (prawa w kodzie)", frameR)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('diagnosis_cam0.png', frameL)
        cv2.imwrite('diagnosis_cam2.png', frameR)
        print("\n" + "=" * 70)
        print("SNAPSHOT ZAPISANY!")
        print("=" * 70)
        print("\nTERAZ ODPOWIEDZ NA PYTANIA:")
        print("\n1. KTÓRA KAMERA WIDZI PIŁKĘ BARDZIEJ PO LEWEJ?")
        print("   (wpisz '0' lub '2')")
        
        which = input("   Odpowiedź: ").strip()
        
        print("\n2. W KTÓREJ KAMERZE PIŁKA JEST BLIŻEJ ŚRODKA?")
        print("   Czyli gdzie X jest NAJBLIŻEJ 320?")
        print("   (wpisz '0' lub '2')")
        
        center_cam = input("   Odpowiedź: ").strip()
        
        print("\n3. PRZYBLIŻONE POZYCJE X:")
        print("   W kamerze INDEX 0 piłka jest przy X ≈ ?")
        x0 = input("   X dla cam0: ").strip()
        
        print("   W kamerze INDEX 2 piłka jest przy X ≈ ?")
        x2 = input("   X dla cam2: ").strip()
        
        # Analiza
        print("\n" + "=" * 70)
        print("ANALIZA:")
        print("=" * 70)
        
        try:
            x0_val = int(x0)
            x2_val = int(x2)
            
            print(f"\nPozycje: cam0={x0_val}, cam2={x2_val}")
            print(f"Różnica: {abs(x0_val - x2_val)}px")
            
            if x0_val > x2_val:
                print("\n✓ cam0 widzi piłkę BARDZIEJ PO PRAWEJ")
                print("  To znaczy że cam0 jest FIZYCZNIE PO LEWEJ")
                print("  Układ jest OK: cam0=LEWA, cam2=PRAWA")
                
                if center_cam == "0":
                    print("\n⚠ ALE cam0 widzi piłkę bliżej środka!")
                    print("  To znaczy że KOREKTA powinna ODEJMOWAĆ offset")
                    print("  (tak jak teraz w kodzie)")
                else:
                    print("\n⚠ A cam2 widzi piłkę bliżej środka!")
                    print("  To dziwne... sprawdź kalibrację!")
                    
            elif x2_val > x0_val:
                print("\n❌ cam2 widzi piłkę BARDZIEJ PO PRAWEJ")
                print("  To znaczy że cam2 jest FIZYCZNIE PO LEWEJ")
                print("  UKŁAD JEST ODWROTNY!")
                print("\n  ROZWIĄZANIE: Zamień kamery w kodzie:")
                print("    capL = cv2.VideoCapture(2)  # LEWA")
                print("    capR = cv2.VideoCapture(0)  # PRAWA")
            else:
                print("\n? Obie kamery pokazują tę samą pozycję X?")
                print("  To niemożliwe dla układu stereo... sprawdź piłkę!")
                
        except:
            print("\n⚠ Nie można przetworzyć wartości X")
        
        print("\n" + "=" * 70)
        print("Naciśnij 'q' aby zakończyć")

capL.release()
capR.release()
cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("PODSUMOWANIE:")
print("=" * 70)
print("\nOBSERWACJE:")
print("1. Kamera która widzi obiekt BARDZIEJ PO LEWEJ = jest FIZYCZNIE po lewej")
print("2. Dla obiektu NA ŚRODKU robota:")
print("   - Lewa kamera widzi go PO PRAWEJ (większe X)")
print("   - Prawa kamera widzi go PO LEWEJ (mniejsze X)")
print("\n3. Piłka NA ŚRODKU robota powinna mieć:")
print("   - X_left > 320")
print("   - X_right < 320")
print("   - Korekta: x_robot = x_left - offset (ODEJMOWANIE)")
print("=" * 70)
