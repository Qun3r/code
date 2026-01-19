import cv2

fs = cv2.FileStorage("stereoMap.xml",cv2.FILE_STORAGE_READ)

P1 = fs.getNode("projMatrixL").mat()
P2 = fs.getNode("projMatrixR").mat()
fx = float(P1[0,0])
Tx = float(P2[0,3])

baseline = abs(Tx /fx)
print("baseline = ",baseline)