import cv2
import os
import pickle
import numpy as np

from utils import align_face


# Cette fonction permet de créer des vecteurs à partir de photos déjà enregistrées
# Les photos doivent être enregistrées dans  le dossier /home/pi/Documents/model_project/photos_persons/NOM_DE_LA_PERSONNE
# avec NOM_DE_LA_PERSONNE à remplacer par le prénom. Idéalement il faut qu'il y ait plusieurs photos par personnes (5 ou 6 au moins)
# il ne doit y avoir qu'un visage sur la photo
def create_vectors(face_detector, facial_landmarks, face_recognition):
	
	path = "/home/pi/Documents/model_project/photos_persons"
	file_vectors = "/home/pi/Documents/model_project/vectors/vectors.pkl"
	file_names = "/home/pi/Documents/model_project/vectors/names.pkl"
	persons = os.listdir(path)
	list_vectors = list()
	list_names = []
	
	# Récupération des dossier de chaque personne
	for person in persons:
		print(person)
		photos = os.listdir(path + "/" + person)
		# Récupération de toutes les photos d'une personne
		for photo in photos:
			print(photo)
			vectors = np.zeros(256)
			# ouverture et redimmensionnement de la photo
			frame = cv2.imread(path + "/" + person + "/" + photo, cv2.IMREAD_COLOR)
			if frame.shape[1] > 640:
				frame = cv2.resize(frame, dsize=None, fx=640/frame.shape[1], fy=640/frame.shape[1])
			initial_w = frame.shape[0]
			initial_h = frame.shape[1]
			
			# detection du visage de la photo
			face_detector.submit_req(frame)
			face = face_detector.inference()[0][0][0]
			xmin = int(face[3] * initial_w)
			ymin = int(face[4] * initial_h)
			xmax = int(face[5] * initial_w)
			ymax = int(face[6] * initial_h)
            
            # Si le visage ne sors pas de la photo
			if xmin>0 and ymin>0 and xmax<initial_w and ymax<initial_h:
				face_frame = frame[ymin:ymax, xmin:xmax]
				# Detection des landmarks du visage
				facial_landmarks.submit_req(face_frame)
				landmarks = facial_landmarks.inference(face_frame)
				# Réaligenement du visage pour qu'il soit droit
				aligned_face = align_face(face_frame, landmarks)
				
				# Génération du vecteur correspondant au visage
				face_recognition.submit_req(aligned_face)
				vectors = face_recognition.inference()
				list_vectors.append(vectors)
				list_names.append(person)
	
	# Sauvegarde de tous les vecteurs dans un fichier pickle
	list_vectors = np.asarray(list_vectors)
	pickle.dump(list_vectors, open(file_vectors, "wb"))
	pickle.dump(list_names, open(file_names, "wb"))
