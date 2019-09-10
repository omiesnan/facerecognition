# ambil library untuk menunjang proses latihan klasifikasi wajah
import cv2
import os
import numpy as np
from PIL import Image
import pickle

# lokasi folder foto untuk latihan klasifikasi wajah
BASE_DIR	=	os.path.dirname(os.path.abspath(__file__))
image_dir	=	os.path.join(BASE_DIR, "images")

# library untuk klasifikasi wajah menggunakan metode haarcascase
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# identitas untuk kllasifikasi wajah
current_id	=	0
label_ids	=	{}
y_labels	=	[]
x_train		=	[]

# klasifikasi
for root, dirs, files in os.walk(image_dir):
	for file in files:
		# format foto yang digunakan untuk latihan dan lokasinya
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path	=	os.path.join(root, file) 
			label 	=	os.path.basename(root).replace(" ", "-").lower()
			print(label, path)
			# proses klasifikasi
			if not label in label_ids:
				label_ids[label] = current_id
				current_id +=     1
				id_ = label_ids[label]
			#print(label_ids)
			#y_labels.append(label)	# some number
			#x_train.append(path) # verify this image, turn into NUMPY array, GRAY
			pil_image = Image.open(path).convert("L") #grayscale
			image_array = np.array(pil_image, "uint8")
			print(image_array)
			faces = face_cascade.detectMultiScale(
				image_array,
				scaleFactor=1.3,
				minNeighbors=5,
				minSize=(30,30)
				)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)

 #print(y_labels)
 #print(x_train)

with open("labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")