import cv2
import json
import time

FRAME_W, FRAME_H = 640, 480
CANDIDATES = list(range(0, 6))
OUT_CFG = "camera_config.json"

def open_caps():
    caps = []
    for idx in CANDIDATES:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        time.sleep(0.1)
        ret, frame = cap.read()
        if ret and frame is not None:
            caps.append((idx, cap))
        else:
            cap.release()
    return caps

caps = open_caps()
if len(caps) < 2:
    print("ERROR: znaleziono mniej niż 2 działające kamery.")
    raise SystemExit

print("Wykryte indeksy:", [i for i, _ in caps])
print("Instrukcja: stań OD TYŁU robota (tak jak jedzie).")
print("Najpierw wybierasz LEWĄ kamerę, potem PRAWĄ.")
print("Naciśnij klawisz 1/2/3/4 dla slotu na ekranie, q = wyjście.")

left_idx = None
right_idx = None

while True:
    frames = []
    for i, cap in caps[:4]:
        ret, f = cap.read()
        frames.append((i, f if ret else None))

    tiles = []
    for k, (i, f) in enumerate(frames, start=1):
        if f is None:
            tile = 255 * (cv2.UMat(FRAME_H, FRAME_W, cv2.CV_8UC3)).get()
        else:
            tile = cv2.resize(f, (FRAME_W, FRAME_H))
        cv2.putText(tile, f"SLOT {k}  index={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)

    while len(tiles) < 4:
        blank = 255 * (cv2.UMat(FRAME_H, FRAME_W, cv2.CV_8UC3)).get()
        tiles.append(blank)

    top = cv2.hconcat(tiles[0:2])
    bot = cv2.hconcat(tiles[2:4])
    grid = cv2.vconcat([top, bot])

    status = f"LEFT={left_idx}  RIGHT={right_idx}"
    cv2.putText(grid, status, (10, FRAME_H * 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("camera_setup (patrz OD TYLU robota)", grid)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key in (ord('1'), ord('2'), ord('3'), ord('4')):
        slot = int(chr(key)) - 1
        if slot >= len(frames):
            continue
        chosen_idx = frames[slot][0]

        if left_idx is None:
            left_idx = chosen_idx
            print("Wybrano LEWA:", left_idx)
        else:
            right_idx = chosen_idx
            print("Wybrano PRAWA:", right_idx)

            cfg = {"left_index": left_idx, "right_index": right_idx, "frame_w": FRAME_W, "frame_h": FRAME_H}
            with open(OUT_CFG, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print("Zapisano:", OUT_CFG, cfg)
            break

for _, cap in caps:
    cap.release()
cv2.destroyAllWindows()
