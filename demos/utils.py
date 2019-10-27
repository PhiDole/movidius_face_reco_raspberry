import cv2
import pickle
import numpy as np
import math

# Cette fonction permet à partir des landmarks du visage de réaligner la tête afin qu'elle soit
# droite et avoir la meilleur reconnaissance faciale possible
def align_face(face, landmarks):
	left_eye = landmarks[0]
	right_eye = landmarks[2]
	dy = right_eye[1] - left_eye[1]
	dx = right_eye[0] - left_eye[0]
	angle = np.arctan2(dy, dx) * 180 /np.pi
	
	center = (face.shape[0]//2, face.shape[1]//2)
	h, w, c = face.shape
	
	M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
	aligned_face = cv2.warpAffine(face, M, (w, h))
	return aligned_face
	
# Calcule la similarité des cosinus afin des matrices des visage pour voir si deux visages sont similaires
def cos_similarity(X, Y):
	Y = Y.T
	return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
	#return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))


# Chargement des vecteurs depuis le fichier pickle
def load_vectors():
	file_vectors = "/home/pi/Documents/model_project/vectors/vectors.pkl"
	file_names = "/home/pi/Documents/model_project/vectors/names.pkl"
	vectors = np.array(pickle.load(open(file_vectors, 'rb')))
	names = np.array(pickle.load(open(file_names, 'rb')))
	
	return vectors, names

# Permet de dessiner les axes pour l'orientation de la tête
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale):
	yaw *= np.pi / 180.0
	pitch *= np.pi / 180.0
	roll *= np.pi / 180.0

	cx = int(center_of_face[0])
	cy = int(center_of_face[1])

	Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                      [0, math.sin(pitch), math.cos(pitch)]])
	Ry = np.array([[math.cos(yaw), 0, -math.sin(yaw)], [0, 1, 0],
                      [math.sin(yaw), 0, math.cos(yaw)]])
	Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                      [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
	R = Rz @ Ry @ Rx  # R = np.dot(Rz, np.dot(Ry, Rx))
	camera_matrix = build_camera_matrix(center_of_face, 950.0)
	xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
	yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
	zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
	zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
	o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
	o[2] = camera_matrix[0][0]

	xaxis = np.dot(R, xaxis) + o
	yaxis = np.dot(R, yaxis) + o
	zaxis = np.dot(R, zaxis) + o
	zaxis1 = np.dot(R, zaxis1) + o

	xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
	yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
	p2 = (int(xp2), int(yp2))
	cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)

	xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
	yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
	p2 = (int(xp2), int(yp2))
	cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)

	xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
	yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
	p1 = (int(xp1), int(yp1))
	xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
	yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
	p2 = (int(xp2), int(yp2))

	cv2.line(frame, p1, p2, (255, 0, 0), 2)
	cv2.circle(frame, p2, 3, (255, 0, 0), 2)

	return frame

def build_camera_matrix(center_of_face, focal_length):
	cx = int(center_of_face[0])
	cy = int(center_of_face[1])
	camera_matrix = np.zeros((3, 3), dtype='float32')
	camera_matrix[0][0] = focal_length
	camera_matrix[0][2] = cx
	camera_matrix[1][1] = focal_length
	camera_matrix[1][2] = cy
	camera_matrix[2][2] = 1
	return camera_matrix
