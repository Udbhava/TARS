import numpy as np
import cv2
import os

haarcascades_path = os.path.join(os.path.dirname(__file__), '') + 'cascades/data/'

face_cascade = cv2.CascadeClassifier(
    haarcascades_path + 'haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=	5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

    cv2.imshow('face', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
