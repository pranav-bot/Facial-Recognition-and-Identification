import cv2
import numpy as np
import pickle
import time

#to classiy face we use cascades
frontal_face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eyes_cascade = cv2.CascadeClassifier(r'C:\Users\prana\OneDrive\Documents\MIt Innovation\facialreg\Face Capture\cascades\data\haarcascade_eye.xml')
eye_glasses = cv2.CascadeClassifier(r'C:\Users\prana\OneDrive\Documents\MIt Innovation\facialreg\Face Capture\cascades\data\haarcascade_eye_tree_eyeglasses.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# To read from the trained model
recognizer.read("trainer.yml")

labels={"Pranav":1}
with open("labels.pickle",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


#To capture video
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #To identify face using cascades
    faces=frontal_face_cascade.detectMultiScale(rgb,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,    y,  w,    h)
        #To capture face we use the cascade co-ordinates
        roi=gray[y:y+h,  x:x+w]
        roi2=frame[y:y+h,  x:x+w]

            #To save image as jpeg
        id_,conf = recognizer.predict(roi)
        if conf>=45 and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_save=(r'C:\Users\prana\OneDrive\Documents\MIt Innovation\facialreg\Face Capture\Captured Images\Pranav\img.jpeg')
        #To write image
        cv2.imwrite(img_save,roi2)


        #To identify face while in live camera
        color=(0,255,0)
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)

    #To show gui of the live camera
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()