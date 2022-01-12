import cv2
import os
from PIL import Image
import numpy as np
import pickle

# Defining the path of the images
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "Captured Images")

# Identifying faces and recognizing them using the previous results from the last run
frontal_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# Adding custom id to all the faces
current_id = 0
label_ids = {}
y_labels = []
x_train = []

# For looking got files and assigning special ids to each of them
for root, dirs, file in os.walk(image_dir):
    for file in file:
        if file.endswith("jpeg") or file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ","-").lower()
            print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids)

            #y_labels.append(label)
            #x_train.append(path)

# To make an array from the cascade values
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = frontal_face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

# To make changes as per the new values learned to the old ones
            for x, y, w, h in faces:
                roi=image_array[y:y+h,x:x+h]
                x_train.append(roi)
                y_labels.append(id_)
print(y_labels)
print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")
