import os
import cv2 as cv
from keras.models import load_model
import numpy as np

model=load_model('mask.model')
face_recog=cv.CascadeClassifier('face_recog/haarcascade_frontalface_default.xml')

video=cv.VideoCapture(0)
target={0: 'mask', 1: 'no mask'}
color={0: (0, 255, 0), 1:(0, 0, 255)}

while True:
    _, img=video.read()
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces=face_recog.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h, in faces:
        face_img=gray[y:y+h, x:x+w]
        face_img=cv.resize(face_img, (125, 125))
        face_img=face_img.astype('float32')/255.0
        face_img=face_img.reshape(-1, 125, 125, 1)

        prediction=model.predict(face_img)[0]
        probability=prediction[np.argmax(prediction)]
        prediction=np.argmax(prediction)
        if probability>0.75:
            cv.rectangle(img, (x, y), (x+w, y+h), color=color[prediction], thickness=2)
            cv.putText(img, target[prediction], (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('video', img)
    key=cv.waitKey(1)
    if key==113:
        break
cv.destroyAllWindows()
video.release()