import cv2
from armv7l.openvino.inference_engine import IENetwork, IEPlugin
import numpy as np

"""Les modèles qui ne sont pas asynchrones, permettent de faire les traitement
principalement pour des images fixes, ici ce sont les photos de personnes à traiter 
pour créer les vecteurs. Ils peuvent être aussi utilisés pour la vidéo mais cela 
rend l'affichage plus lent et moins fluide"""

class FaceDetection(object):
	def __init__(self, plugin):
		# Chargement du modèle à partir du fichier xml et binaire
		model_bin="/home/pi/Documents/model_project/intel_models/face_detection_adas/FP16/face-detection-adas-0001.bin"
		model_xml="/home/pi/Documents/model_project/intel_models/face_detection_adas/FP16/face-detection-adas-0001.xml"
		net=IENetwork(model=model_xml, weights=model_bin)
		# Récupération du nom de la couche d'entrée et de sortie du modèle
		self.input_blob = next(iter(net.inputs))
		self.out_blob = next(iter(net.outputs))
		# Chargement du modèle sur le movidius
		self.exec_net = plugin.load(network=net, num_requests=2)
		# Récupération du format de l'entrée et de la sortie du modèle
		self.input_dims = net.inputs[self.input_blob].shape
		self.outout_dims = net.outputs[self.out_blob].shape
		del net
		
	def submit_req(self, in_frame):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(in_frame, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.infer(inputs={self.input_blob:frame})
	
	def inference(self):
		# Récupération des résultats de l'inférence
		faces = None
		res = self.exec_net.requests[0].outputs[self.out_blob]
		faces = res[0][:, np.where(res[0][0][:, 2] > 0.5)]
		return faces	
		
			
class FacialLandmarks(object):
	def __init__(self, plugin):
		# Chargement du modèle à partir du fichier xml et binaire
		model_bin="/home/pi/Documents/model_project/intel_models/facial_landmarks/FP16/facial-landmarks-35-adas-0002.bin"
		model_xml="/home/pi/Documents/model_project/intel_models/facial_landmarks/FP16/facial-landmarks-35-adas-0002.xml"
		net=IENetwork(model=model_xml, weights=model_bin)
		# Récupération du nom de la couche d'entrée et de sortie du modèle
		self.input_blob = next(iter(net.inputs))
		self.out_blob = next(iter(net.outputs))
		# Chargement du modèle sur le movidius
		self.exec_net = plugin.load(network=net, num_requests=2)
		# Récupération du format de l'entrée et de la sortie du modèle
		self.input_dims = net.inputs[self.input_blob].shape
		self.outout_dims = net.outputs[self.out_blob].shape
		del net
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.infer(inputs={self.input_blob:frame})

			
	def inference(self, face):
		# Récupération des résultats de l'inférence
		res = self.exec_net.requests[0].outputs[self.out_blob]
		res = res.reshape(1, 70)[0]
		
		facial_landmarks = np.zeros((35,2))
		for i in range(res.size //2):
			normed_x = res[2*i]
			normed_y = res[2*i +1]
			x_lm = face.shape[1] * normed_x
			y_lm = face.shape[0] * normed_y
			facial_landmarks[i] = (x_lm, y_lm)
		
		return facial_landmarks


class FaceRecognition(object):
	def __init__(self, plugin):
		# Chargement du modèle à partir du fichier xml et binaire
		model_bin="/home/pi/Documents/model_project/intel_models/face_reidentification/FP16/face-reidentification-retail-0095.bin"
		model_xml="/home/pi/Documents/model_project/intel_models/face_reidentification/FP16/face-reidentification-retail-0095.xml"
		net=IENetwork(model=model_xml, weights=model_bin)
		# Récupération du nom de la couche d'entrée et de sortie du modèle
		self.input_blob = next(iter(net.inputs))
		self.out_blob = next(iter(net.outputs))
		# Chargement du modèle sur le movidius
		self.exec_net = plugin.load(network=net, num_requests=2)
		# Récupération du format de l'entrée et de la sortie du modèle
		self.input_dims = net.inputs[self.input_blob].shape
		self.outout_dims = net.outputs[self.out_blob].shape
		del net
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.infer(inputs={self.input_blob:frame})
		
	
	def inference(self):
		# Récupération des résultats de l'inférence
		res = self.exec_net.requests[0].outputs[self.out_blob]
		face_vector = res.reshape(1, 256)
		return face_vector
