import serial
import time

try:
    # Używamy nowo odkrytego portu
    ser = serial.Serial('/dev/serial0', 9600, timeout=1)
    ser.flush()
    print("Połączono z portem /dev/serial0")

    while True:
        ser.write(b'F')
        print("Wysłano: F")
        time.sleep(2)
except Exception as e:
    print(f"Błąd: {e}")
finally:
    if 'ser' in locals(): ser.close()