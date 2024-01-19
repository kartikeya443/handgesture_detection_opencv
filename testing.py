# dependencies
import cv2
import mediapipe as mp
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


# defined variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
labels = ["A", "B", "C", "Hello!", "Love You!", "No!", "Yes!"]

# data trained using Teachable Machine by withGoogle
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

#function
while True:
    
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCropShape = imgCrop.shape
        
        aspectRatio = h/w
        
        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
            
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        
        cv2.putText(imgOutput, labels[index], (x+127, y-27), cv2.FONT_HERSHEY_COMPLEX, -1.7, (255,255,255), 2, 1, True)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (0,128,0),2)
        
        #cv2.imshow("Cropped Image", imgCrop)
        imgWh = cv2.flip(imgWhite, 1)
        cv2.imshow("Gesture Mapping", imgWh)
    
    
    imgOut = cv2.flip(imgOutput, 1)    
    cv2.imshow("My Camera", imgOut)
    cv2.waitKey(1)