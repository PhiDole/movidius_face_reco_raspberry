import cv2
import numpy as np
from utils import align_face, cos_similarity, draw_axes

# Cette fonction permet de faire la reconnaissance d'age, genre, emotion en direct de façon asynchrone
def live_face_attributs(button, frame, cap, face_detector, age_gender_detection, emotion_detection):
    # Changement d'état du cache en fonction de la detection du gpio
    if button.is_pressed:
        cache = True
    else:
        cache = False
    ret, next_frame = cap.read()
        
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Si le caches est mis on crée une image grise, sinon on récupère la frame actuelle
    if cache:
        new_frame = np.full((int(initial_h), int(initial_w), 3), 100, np.uint8)
    else:
        new_frame = frame
    
    # Detection des visages    
    face_detector.submit_req(next_frame)
    ret = face_detector.wait()
    faces = face_detector.inference()[0][0]
    
    # Pour chaque visage détecté
    for face in faces:
        xmin = int(face[3] * initial_w)
        ymin = int(face[4] * initial_h)
        xmax = int(face[5] * initial_w)
        ymax = int(face[6] * initial_h)
        color = (0, 0, 255)
            
        # Si le visage ne sors pas du cadre
        if xmin>0 and ymin>0 and xmax<initial_w and ymax<initial_h and xmax>0 and ymax>0 and xmin<initial_w and ymin<initial_h :
            # Dessin du carré autour du visage sur l'image
            cv2.rectangle(new_frame, (xmin, ymin), (xmax, ymax), color, 2)
            # On recadre l'image au niveau du visage
            face_frame = frame[ymin:ymax, xmin:xmax]
            # detection de l'émotion et de age et genre
            age_gender_detection.submit_req(face_frame)
            ret = age_gender_detection.wait()
            age, gender = age_gender_detection.inference()
                    
            emotion_detection.submit_req(face_frame)
            ret = emotion_detection.wait()
            emotion = emotion_detection.inference()
            
            # Dessin du résultat sur l'image
            cv2.putText(new_frame, gender + ' ' + str(age) + ' ' + emotion, (xmin, ymin - 7),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        
    frame = next_frame

    return frame, new_frame
        

# Cette fonction permet de faire la reconnaissance faciale en direct de façon asynchrone
def live_face_recognition(button, frame, cap, photo_vectors, names, face_detector, facial_landmarks, face_recognition):
    # Changement d'état du cache en fonction de la detection du gpio
    if button.is_pressed:
        cache = True
    else:
        cache = False
    ret, next_frame = cap.read()

    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Si le caches est mis on crée une image grise, sinon on récupère la frame actuelle
    if cache:
        new_frame = np.full((int(initial_h), int(initial_w), 3), 100, np.uint8)
    else:
        new_frame = frame
    # Detection des visages   
    face_detector.submit_req(next_frame)
    ret = face_detector.wait()
    faces = face_detector.inference()[0][0]
    # Pour chaque visage détecté
    for face in faces:
        xmin = int(face[3] * initial_w)
        ymin = int(face[4] * initial_h)
        xmax = int(face[5] * initial_w)
        ymax = int(face[6] * initial_h)
        color = (0, 0, 255)
            
        # Si le visage ne sors pas du cadre
        if xmin>0 and ymin>0 and xmax<initial_w and ymax<initial_h and xmax>0 and ymax>0 and xmin<initial_w and ymin<initial_h :
            # Dessin du carré autour du visage sur l'image
            cv2.rectangle(new_frame, (xmin, ymin), (xmax, ymax), color, 2)
            # On recadre l'image au niveau du visage
            face_frame = frame[ymin:ymax, xmin:xmax]
                 
            # Reconnaissance des landmarks pour le visage
            facial_landmarks.submit_req(face_frame)
            ret = facial_landmarks.wait()
            landmarks = facial_landmarks.inference(face_frame)
            # Alignement du visage pour qu'il soit droit
            aligned_face = align_face(face_frame, landmarks)
            # Génération du vecteur du visage
            face_recognition.submit_req(aligned_face)
            ret = face_recognition.wait()
            result_vector = face_recognition.inference()
            similarity = []
            # Comparaison du vecteur avec ceux des autres visages
            for vector in photo_vectors:
                sim = cos_similarity(vector, result_vector)
                similarity.append(sim)
				
				
            face_id = np.asarray(similarity).argmax()
            txt= "???"
            # Si le vecteur le plus proche a une distance inférieure au threshold
            # on peut reconnaitre la personne
            if similarity[face_id] < 0.1:
                txt = str(names[face_id])
            # Affichage du résultat sur l'image
            cv2.putText(new_frame, txt, (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    frame = next_frame
    return frame, new_frame

# Cette fonction permet de faire la reconnaissance des landmarks et de l'angle de la tête en direct de façon asynchrone
def live_face_details(button, frame, cap, face_detector, facial_landmarks, head_pose):
    # Changement d'état du cache en fonction de la detection du gpio
    if button.is_pressed:
        cache = True
    else:
        cache = False
        
    ret, next_frame = cap.read()
        
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    # Si le caches est mis on crée une image grise, sinon on récupère la frame actuelle
    if cache:
        new_frame = np.full((int(initial_h), int(initial_w), 3), 100, np.uint8)
    else:
        new_frame = frame
    # Detection des visages
    face_detector.submit_req(next_frame)
    ret = face_detector.wait()
    faces = face_detector.inference()[0][0]
    # Pour chaque visage détecté
    for face in faces:
        xmin = int(face[3] * initial_w)
        ymin = int(face[4] * initial_h)
        xmax = int(face[5] * initial_w)
        ymax = int(face[6] * initial_h)
        color = (0, 0, 255)
            
        # Si le visage ne sors pas du cadre
        if xmin>0 and ymin>0 and xmax<initial_w and ymax<initial_h and xmax>0 and ymax>0 and xmin<initial_w and ymin<initial_h :
            # On recadre l'image au niveau du visage
            face_frame = frame[ymin:ymax, xmin:xmax]
            face_w, face_h = face_frame.shape[:2]
            # Detection de la position de la tête
            head_pose.submit_req(face_frame)
            ret = head_pose.wait()
            yaw, pitch, roll = head_pose.inference()
            # Dessin des axes de la tête
            if face_h != 0 and face_w != 0:
                    center_of_face = (xmin + face_h / 2, ymin + face_w / 2, 0)
                    new_frame = draw_axes(new_frame, center_of_face, yaw, pitch, roll, 50)

            # Detection des landmarks
            facial_landmarks.submit_req(face_frame)
            ret = facial_landmarks.wait()
            normed_landmarks = facial_landmarks.inference(face_frame)
            # Dessin des landmarks
            n_lm = normed_landmarks.size
            for i in range(int(n_lm/2)):
                normed_x = normed_landmarks[i][0]
                normed_y = normed_landmarks[i][1]
                x_lm = xmin + normed_x
                y_lm = ymin + normed_y
                if x_lm > 0 and y_lm > 0 and x_lm < initial_w and y_lm<initial_h:
                    a = 1 + int(0.012 * face_h)
                    cv2.circle(new_frame, (int(x_lm), int(y_lm)), a, (0, 255, 255), -1)
        
    frame = next_frame

    return frame, new_frame

		
