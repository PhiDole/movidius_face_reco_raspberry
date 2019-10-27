import cv2
import os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from gpiozero import Button

from armv7l.openvino.inference_engine import IEPlugin

from load_model_async import FaceDetection, AgeGenderDetection, EmotionDetection, FacialLandmarks, FaceRecognition, HeadPoseDetection
from load_model_notasync import FaceDetection as FaceDetectionSync
from load_model_notasync import FacialLandmarks as FacialLandmarksSync
from load_model_notasync import FaceRecognition as FaceRecognitionSync

from preprocessing import create_vectors
from utils import load_vectors
from live_demo import live_face_attributs, live_face_recognition, live_face_details

# Pour changer la taille de l'image, plus l'image est grand, plus le temps de traitement est long
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Création d'une interface graphique
root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()


# Fonction qui est lancée avant l'affichage de la démo et qui permet de 
# charger tous les modèles de réseau de neurones sur le movidius
# il permet aussi de créer le bouton lié au GPIO, pour changer l'image 
# si le cache est devant la caméra ou pas

def init_algo():
	global algo, button, frame, vectors, names, face_detector, age_gender, emotion, facial_landmarks, face_recognition, head_pose
	# algo permet de définir quel algo est en train de tourner, soit la reconnaissance faciale,
	# soit la détection age, genre, emotion, soit la reconnaissance des détails du visages (landmarks et orientation de la tete)
	algo ="None"
	button = Button(14)
	
	print("load models...")
	# Chargement du plugin pour tous les réseaux de neurones
	plugin = IEPlugin(device="MYRIAD", plugin_dirs=None)
	
	# Chargement de tous les modèles
	face_detector = FaceDetection(plugin)
	age_gender = AgeGenderDetection(plugin)
	emotion = EmotionDetection(plugin)
	facial_landmarks = FacialLandmarks(plugin)
	head_pose = HeadPoseDetection(plugin)
	face_recognition = FaceRecognition(plugin)
	
	face_detector_sync = FaceDetectionSync(plugin)
	facial_landmarks_sync = FacialLandmarksSync(plugin)
	face_recognition_sync = FaceRecognitionSync(plugin)
	
	# Création des vecteurs pour les photos de personnes déjà enregistrées
	print("create vectors for face recognition...")
	create_vectors(face_detector_sync, facial_landmarks_sync, face_recognition_sync)
	vectors, names = load_vectors()	
	
	# Récupération de la première image pour faire la detection en asynchrone
	print("start recording")
	_, frame = cap.read()

# Fonction qui se lance quand l'utilisateur appuie sur "face attibuts" et qui définie
# que l'algo à faire tourner est celui pour la reconnaissance d'age, genre, emotion
def set_face_attibuts(event=0):
	global algo
	algo = "face_attributs"

# Fonction qui se lance quand l'utilisateur appuie sur "face recognition" et qui définie
# que l'algo à faire tourner est celui pour la reconnaissance de personne
def set_face_recognition(event=0):
	global algo
	algo = "face_recognition"

# Fonction qui se lance quand l'utilisateur appuie sur "face details" et qui définie
# que l'algo à faire tourner est celui pour la reconnaissance de landmarks et d'orientation de la tête
def set_face_details(event=0):
	global algo
	algo = "face_details"
    
# Fonction qui est appelé pour chaque image de la vidéo et qui permet de lancer les algos en fonction 
# de celui qui est selectionné
def show_frame():
	global frame
	if algo is "face_attributs":
		frame, new_frame=live_face_attributs(button, frame, cap, face_detector, age_gender, emotion)
	elif algo is "face_recognition":
		frame, new_frame=live_face_recognition(button, frame, cap, vectors, names, face_detector, facial_landmarks, face_recognition)
	elif algo is "face_details":
		frame, new_frame=live_face_details(button, frame, cap, face_detector, facial_landmarks, head_pose)
	else:
		# lorsque qu'aucun algo est selctionné on affiche une image noire
		new_frame = np.full((int(height), int(width), 3), 10, np.uint8)
	# On affiche l'image traité sur l'interface graphique
	cv2image = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGBA)
	img = Image.fromarray(cv2image)
	imgtk = ImageTk.PhotoImage(image=img)

	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(10, show_frame)

# Définition des bouttons à cliquer sur l'interface pour le choix de l'algo
button1 = tk.Button(root, text="face attributes", command=set_face_attibuts)
button1.pack(side=tk.LEFT, padx=5, pady=5)
button2 = tk.Button(root, text="face recognition", command=set_face_recognition)
button2.pack(side=tk.LEFT, padx=5, pady=5)
button3 = tk.Button(root, text="face details", command=set_face_details)
button3.pack(side=tk.LEFT, padx=5, pady=5)

# On lance l'initialisation puis le programme
init_algo()
show_frame()
root.mainloop()
