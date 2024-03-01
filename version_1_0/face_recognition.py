#pylint:disable=no-member

import numpy as np
import cv2

haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('./face_trained.yml')

img = cv2.imread('./Ressources/Faces/val/elton_john/1.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Person', gray)

#Dectect Face in the image
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
for (x,y,w,h) in face_rect:
    faces_roi = gray[y:y+w,x:x+w]
    
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} avec une confiance de {confidence}')
    
    cv2.putText(img, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
    
cv2.imshow('Detected Face', img)
cv2.waitKey(0)
cv2.destroyAllWindows()