import os
import cv2
import numpy as np
from keras.models import model_from_json
from model import FacialExpressionModel
from keras.preprocessing import image
from model import FacialExpressionModel

#load model
model = FacialExpressionModel("model.json", "model_weights.h5")
#load weights



face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

while True:
    ret,fr=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_fr= cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
   
    faces = face_haar_cascade.detectMultiScale(gray_fr, 1.32, 5)


    for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
    resized_img = cv2.resize(fr, (1000, 700))        
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows