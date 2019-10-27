import cv2
from armv7l.openvino.inference_engine import IENetwork, IEPlugin
import numpy as np

""" Les modèles asynchrones permettent de faire l'inférence pour des vidéos en
en traitant l'image avant celle actuellement récupérée (traitement de l'image à t-1)
Cela permet de rendre la vidéo plus fluide."""

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
		self.output_dims = net.outputs[self.out_blob].shape
		del net
		
		# première mise à jour de l'état pour l'asynchronicité
		self.cur_request_id = 0
		self.next_request_id = 1
		
	def submit_req(self, in_frame):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(in_frame, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob:frame})
		
	def wait(self):
		# Attente en fonction de l'état de l'aynchronicité
		if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
			return True
		else:
			return False
	
	def inference(self):
		# Récupération des résultats de l'inférence
		faces = None
		res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
		faces = res[0][:, np.where(res[0][0][:, 2] > 0.5)]
		# mise à jour de l'état de l'asynchronicité
		self.cur_resquest_id, self.next_request_id = self.next_request_id, self.cur_request_id
		return faces	
		

class AgeGenderDetection(object):
	def __init__(self, plugin):
		# Chargement du modèle à partir du fichier xml et binaire
		model_bin="/home/pi/Documents/model_project/intel_models/age_gender_recognition_retail/FP16/age-gender-recognition-retail-0013.bin"
		model_xml="/home/pi/Documents/model_project/intel_models/age_gender_recognition_retail/FP16/age-gender-recognition-retail-0013.xml"
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
		# Création des labels
		self.label = ('Woman', 'Man')
		# première mise à jour de l'état pour l'asynchronicité
		self.cur_request_id = 0
		self.next_request_id = 1
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob:frame})
		
	def wait(self):
		# Attente en fonction de l'état de l'aynchronicité
		if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
			return True
		else:
			return False
		
	
	def inference(self):
		# Récupération des résultats de l'inférence
		age=0
		gender=""
		age = self.exec_net.requests[self.cur_request_id].outputs['age_conv3']
		prob = self.exec_net.requests[self.cur_request_id].outputs['prob']
		age = age[0][0][0][0] * 100
		if prob[0][0] > 0.2:
			gender = "woman"
		else:
			gender = "man"
		# mise à jour de l'état de l'asynchronicité	
		self.cur_resquest_id, self.next_request_id = self.next_request_id, self.cur_request_id
		
		return int(age), gender
		

class EmotionDetection(object):
	def __init__(self, plugin):
		# Chargement du modèle à partir du fichier xml et binaire
		model_bin="/home/pi/Documents/model_project/intel_models/emotions_recognition_retail/FP16/emotions-recognition-retail-0003.bin"
		model_xml="/home/pi/Documents/model_project/intel_models/emotions_recognition_retail/FP16/emotions-recognition-retail-0003.xml"
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
		# Création des labels
		self.label = ('neutral', 'happy', 'sad', 'surprise', 'anger')
		# première mise à jour de l'état pour l'asynchronicité
		self.cur_request_id = 0
		self.next_request_id = 1
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob:frame})
		
	def wait(self):
		# Attente en fonction de l'état de l'aynchronicité
		if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
			return True
		else:
			return False
		
	
	def inference(self):
		# Récupération des résultats de l'inférence
		emotion=""
		res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
		emotion = self.label[np.argmax(res[0])]
		# mise à jour de l'état de l'asynchronicité
		self.cur_resquest_id, self.next_request_id = self.next_request_id, self.cur_request_id
		
		return emotion
		
				
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
		# première mise à jour de l'état pour l'asynchronicité
		self.cur_request_id = 0
		self.next_request_id = 1
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob:frame})
		
	def wait(self):
		# Attente en fonction de l'état de l'aynchronicité
		if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
			return True
		else:
			return False
			
	def inference(self, face):
		# Récupération des résultats de l'inférence
		res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
		res = res.reshape(1, 70)[0]
		
		facial_landmarks = np.zeros((35,2))
		for i in range(res.size //2):
			normed_x = res[2*i]
			normed_y = res[2*i +1]
			x_lm = face.shape[1] * normed_x
			y_lm = face.shape[0] * normed_y
			facial_landmarks[i] = (x_lm, y_lm)
		# mise à jour de l'état de l'asynchronicité
		self.cur_resquest_id, self.next_request_id = self.next_request_id, self.cur_request_id
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
		# première mise à jour de l'état pour l'asynchronicité
		self.cur_request_id = 0
		self.next_request_id = 1
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob:frame})
		
	def wait(self):
		# Attente en fonction de l'état de l'aynchronicité
		if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
			return True
		else:
			return False
		
	
	def inference(self):
		# Récupération des résultats de l'inférence
		res = self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
		face_vector = res.reshape(1, 256)
		# mise à jour de l'état de l'asynchronicité
		self.cur_resquest_id, self.next_request_id = self.next_request_id, self.cur_request_id
		
		return face_vector
		
		
class HeadPoseDetection(object):
	def __init__(self, plugin):
		# Chargement du modèle à partir du fichier xml et binaire
		model_bin="/home/pi/Documents/model_project/intel_models/head_pose_estimation/FP16/head-pose-estimation-adas-0001.bin"
		model_xml="/home/pi/Documents/model_project/intel_models/head_pose_estimation/FP16/head-pose-estimation-adas-0001.xml"
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
		# première mise à jour de l'état pour l'asynchronicité
		self.cur_request_id = 0
		self.next_request_id = 1
		
	def submit_req(self, face):
		# Traitement de l'image et passage dans le réseau de neurones
		n, c, h, w = self.input_dims
		frame = cv2.resize(face, (w, h))
		frame = frame.transpose((2, 0, 1))
		frame = frame.reshape((n,c,h,w))
		self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob:frame})
		
	def wait(self):
		# Attente en fonction de l'état de l'aynchronicité
		if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
			return True
		else:
			return False
		
	
	def inference(self):
		# Récupération des résultats de l'inférence
		yaw = 0 # z
		pitch = 0 # y
		roll = 0 # x
		yaw = self.exec_net.requests[self.cur_request_id].outputs['angle_y_fc'][0][0]
		pitch = self.exec_net.requests[self.cur_request_id].outputs['angle_y_fc'][0][0]
		roll = self.exec_net.requests[self.cur_request_id].outputs['angle_y_fc'][0][0]
		# mise à jour de l'état de l'asynchronicité
		self.cur_resquest_id, self.next_request_id = self.next_request_id, self.cur_request_id
		
		return yaw, pitch, roll

		
	
		

