import cv2
import numpy as np
import pickle
import tkinter
from tkinter import messagebox

# gunakan library opencv haarcascade
faceCascade	 =	cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer   =  cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
title_orangasing = "Peringatan!"
body_orangasing = "Siapa yang ada di depan rumah? Cek dong!"

# tampilkan nama pemilik wajah
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# tangkap gambar dari webcam
video_capture = cv2.VideoCapture(0)

while True :

	# tangkap gambar frame demi frame dari webcam
	ret, frame	= video_capture.read()
	gray		= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces       = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
	)

	for (x, y, w, h) in faces:
		#print(x,y,w,h)
		roi_gray	=	gray[y:y+h, x:x+w]
		roi_color	=	frame[y:y+h, x:x+w]

		# pengenalan dan prediksi pemilik wajah.. seberapa akurat?
		id_, conf	=	recognizer.predict(roi_gray)
		if conf >= 15 and conf <= 85: #jika
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		else:
			print('Orang_asing')
			messagebox.showerror(title_orangasing, body_orangasing)

		# simpan hasil deteksi wajah
		img_item	=	"my.image.png"
		cv2.imwrite(img_item, roi_gray)

		# buat kotak sebagai penanda tracking wajah
		color 		=	(255, 0, 0) 
		stroke		=	2
		end_core_x = x + w
		end_core_y = y + h
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
 	#tampikan rekaman langsung dari webcam
	cv2.imshow('Smart Surveillance - Face Detection', frame)
	if cv2.waitKey(1) & 0xFF == ord ('e'):
		break

#setelah semua selesai, tutup semua jendela
video_capture.release()
cv2.destroyAllWindows()