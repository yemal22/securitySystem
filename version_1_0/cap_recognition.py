#pylint:disable=no-member

import numpy as np
import cv2

haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling', 'Morel KPAVODE']

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('./face_trained.yml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Dectect Face in the image
    face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x,y,w,h) in face_rect:
        faces_roi = gray[y:y+w,x:x+w]
    
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} avec une confiance de {confidence}')
    
        cv2.putText(frame, str(people[label]), (x-20,y-20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
    
    cv2.imshow('Webcam', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
