import cv2
import numpy as np
import time
import os
import HandTrackModule as htm

folderPath = "Header"
mylist = os.listdir(folderPath)
overlayList = []

for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
drawColor = (255,255,255)
header = overlayList[0]
brushThick = 20

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)
xprev,yprev = 0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw = False)
    if len(lmList) != 0 :
        x1,y1 = lmList[8][1],lmList[8][2]
        x2,y2 = lmList[12][1],lmList[12][2]
        fingers = detector.fingersUp() 
        if fingers[1]==1 and fingers[2]==1:
            print("Selection Mode")
            
            if y1<144:
                if 100<x1<350:
                    header = overlayList[1]
                    drawColor  = (0,0,255)
                    brushThick = 10

                elif 450<x1<750:
                    header = overlayList[2]
                    drawColor  = (255,0,0)
                    brushThick = 10

                elif 800<x1<950:
                    header = overlayList[3]
                    drawColor  = (0,255,0)
                    brushThick = 10
                elif 1050<x1<1200:
                    header = overlayList[4] 
                    drawColor  = (0,0,0)
                    brushThick = 250
            cv2.circle(img,(x1,y1),25,drawColor,cv2.FILLED)
        elif fingers[1]==1 and fingers[2]==0:
            print("Drawing mode")
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            if xprev ==0 and yprev ==0:
                xprev,yprev = x1,y1
            cv2.line(img,(xprev,yprev),(x1,y1),drawColor,20)
            cv2.line(imgCanvas,(xprev,yprev),(x1,y1),drawColor,brushThick)
            xprev,yprev = x1,y1
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    
    img[0:144,0:1280] = header

    cv2.imshow("Image",img)
    cv2.imshow("Inverse",imgInv)
    cv2.imshow("Image canvas",imgCanvas)
    cv2.waitKey(1)