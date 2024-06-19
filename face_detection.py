
import dlib
import numpy as np
import os
import cv2

pose_predictor = dlib.shape_predictor('dependecies/posePredictor.dat')
face_encoder = dlib.face_recognition_model_v1('dependecies/128Emb.dat')

face_detector = dlib.get_frontal_face_detector()

def get_face(image):
    faces = face_detector(image, 1)
    if faces:
        return faces[0]
    return None

def encode_face(image):
    face_location = get_face(image)
    if face_location is None:
        return None
    face_landmarks = pose_predictor(image, face_location)
    face_chip = dlib.get_face_chip(image, face_landmarks)
    encodings = np.array(face_encoder.compute_face_descriptor(face_chip))
    return encodings, face_location

def get_similarity(encoding1, encoding2):
    return np.linalg.norm(encoding1 - encoding2)

def compare_encoded_faces(encoding1, encoding2):
    if encoding1 is None or encoding2 is None:
        return False
    distance = get_similarity(encoding1, encoding2)
    return distance < 0.6

def save_cropped_face(image, face_location, save_dir):
    top, right, bottom, left = (face_location.top(), face_location.right(), face_location.bottom(), face_location.left())
    face_image = image[top:bottom, left:right]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    face_path = os.path.join(save_dir, f"face_{len(os.listdir(save_dir)) + 1}.jpg")
    cv2.imwrite(face_path, face_image)
    return face_path