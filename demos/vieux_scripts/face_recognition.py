import cv2
import os
import pickle
import numpy as np

from utils import align_face

#from load_model_async import FaceDetection, FacialLandmarks, FaceRecognition
#from armv7l.openvino.inference_engine import IENetwork, IEPlugin
#from gpiozero import Button


"""
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

def cos_similarity(X, Y):
	Y = Y.T
	return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
	#return np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))


def load_vectors():
	file_vectors = "/home/pi/Documents/model_project/vectors/vectors.pkl"
	file_names = "/home/pi/Documents/model_project/vectors/names.pkl"
	vectors = np.array(pickle.load(open(file_vectors, 'rb')))
	names = np.array(pickle.load(open(file_names, 'rb')))
	
	return vectors, names
	"""

def preprocess_images(face_detector, facial_landmarks, face_recognition):
	
	path = "/home/pi/Documents/model_project/photos_persons"
	file_vectors = "/home/pi/Documents/model_project/vectors/vectors.pkl"
	file_names = "/home/pi/Documents/model_project/vectors/names.pkl"
	persons = os.listdir(path)
	list_vectors = list()
	list_names = []
	for person in persons:
		print(person)
		photos = os.listdir(path + "/" + person)
		for photo in photos:
			print(photo)
			vectors = np.zeros(256)
			
			frame = cv2.imread(path + "/" + person + "/" + photo, cv2.IMREAD_COLOR)
			if frame.shape[1] > 640:
				frame = cv2.resize(frame, dsize=None, fx=640/frame.shape[1], fy=640/frame.shape[1])
			initial_w = frame.shape[0]
			initial_h = frame.shape[1]
			face_detector.submit_req(frame)
			face = face_detector.inference()[0][0][0]
			xmin = int(face[3] * initial_w)
			ymin = int(face[4] * initial_h)
			xmax = int(face[5] * initial_w)
			ymax = int(face[6] * initial_h)
            
			if xmin>0 and ymin>0 and xmax<initial_w and ymax<initial_h:
				face_frame = frame[ymin:ymax, xmin:xmax]
				facial_landmarks.submit_req(face_frame)
				landmarks = facial_landmarks.inference(face_frame)
				aligned_face = align_face(face_frame, landmarks)
				face_recognition.submit_req(aligned_face)
				vectors = face_recognition.inference()
				list_vectors.append(vectors)
				list_names.append(person)
	
	list_vectors = np.asarray(list_vectors)
	pickle.dump(list_vectors, open(file_vectors, "wb"))
	pickle.dump(list_names, open(file_names, "wb"))
	
"""
def live_face_reco(photo_vectors, names, face_detector, facial_landmarks, face_recognition):
	button1 = Button(14)
	button2 = Button(15) 
    
	if button1.is_pressed:
		cache=True
	else:
		cache=False
   
	input_stream = 0

	cap = cv2.VideoCapture(input_stream)

	ret, frame = cap.read()

	print("To close the application, press 'CTRL+C' or any key with focus on the output window")
	while cap.isOpened():
		if button1.is_pressed:
			cache = True
		else:
			cache = False
		ret, next_frame = cap.read()
        

		if not ret:
			break
		initial_w = cap.get(3)
		initial_h = cap.get(4)

        
		if cache:
			new_frame = np.full((int(initial_h), int(initial_w), 3), 100, np.uint8)
		else:
			new_frame = frame
        
		face_detector.submit_req(next_frame)
		ret = face_detector.wait()
		faces = face_detector.inference()[0][0]
		for face in faces:
			xmin = int(face[3] * initial_w)
			ymin = int(face[4] * initial_h)
			xmax = int(face[5] * initial_w)
			ymax = int(face[6] * initial_h)
			color = (0, 0, 255)
            
            
			if xmin>0 and ymin>0 and xmax<initial_w and ymax<initial_h and xmax>0 and ymax>0 and xmin<initial_w and ymin<initial_h :
				cv2.rectangle(new_frame, (xmin, ymin), (xmax, ymax), color, 2)
				face_frame = frame[ymin:ymax, xmin:xmax]
                 
    
				facial_landmarks.submit_req(face_frame)
				ret = facial_landmarks.wait()
				landmarks = facial_landmarks.inference(face_frame)
				aligned_face = align_face(face_frame, landmarks)
				face_recognition.submit_req(aligned_face)
				ret = face_recognition.wait()
				result_vector = face_recognition.inference()
				similarity = []
				for vector in photo_vectors:
					sim = cos_similarity(vector, result_vector)
					similarity.append(sim)
				
				
				face_id = np.asarray(similarity).argmax()
				txt= "???"
				print(similarity[face_id])
				if similarity[face_id] < 0.1:
					txt = str(names[face_id])
				cv2.putText(new_frame, txt, (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)


		cv2.imshow("Detection Results", new_frame)

		frame = next_frame

		key = cv2.waitKey(1)
		if key == 27:
			break

	cv2.destroyAllWindows()
	
	
plugin = IEPlugin(device="MYRIAD", plugin_dirs=None)
print("load models")
face_detector = FaceDetection(plugin)
facial_landmarks = FacialLandmarks(plugin)
face_recognition = FaceRecognition(plugin)
#preprocess_images(face_detector, facial_landmarks, face_recognition)
photo_vectors, names = load_vectors()
print(names)
live_face_reco(photo_vectors, names, face_detector, facial_landmarks, face_recognition)
print("fini!")
"""

