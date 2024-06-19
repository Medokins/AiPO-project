import os
import cv2
import pandas as pd
from tkinter import Tk, filedialog
from text_extraction import preprocess_image, recognize_text, extract_information
from face_detection import encode_face, compare_encoded_faces, save_cropped_face
import matplotlib.pyplot as plt


def load_image():
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename()
    return image_path

def read_image(image_path):
    return cv2.imread(image_path)

def update_database(image_path, name, surname):
    database_path = 'database.csv'
    save_dir = 'detected_faces'

    if os.path.exists(database_path):
        database = pd.read_csv(database_path)
    else:
        database = pd.DataFrame(columns=["name", "surname", "path_to_face"])

    new_face_encoding, new_face_location = encode_face(read_image(image_path))
    if new_face_encoding is None:
        print("No face detected in the image.")
        return

    face_exists = False
    for index, row in database.iterrows():
        existing_face_encoding, _ = encode_face(cv2.imread(row['path_to_face']))
        if compare_encoded_faces(new_face_encoding, existing_face_encoding):
            print("\nThis person is already in database!\nInformation about this person:\n", 40*"-")
            face_exists = True
            if not pd.isna(row['name']):
                name = row['name']
            if not pd.isna(row['surname']):
                surname = row['surname']
            database.at[index, 'name'] = name
            database.at[index, 'surname'] = surname
            print(f"Name: {name}, surname: {surname}")
            break

    if not face_exists:
        print("This face is not in database!")
        face_path = save_cropped_face(read_image(image_path), new_face_location, save_dir)
        new_record = {
            "name": name,
            "surname": surname,
            "path_to_face": face_path
        }
        database = database._append(new_record, ignore_index=True)

    database.to_csv(database_path, index=False)



if __name__ == "__main__":
    image_path = load_image()
    preprocessed_image = preprocess_image(image_path)
    words = recognize_text(preprocessed_image)

    name, surname, special_number = extract_information(words)
    if name:
        print(f"Extracted Name: {name}")
    if surname:
        print(f"Extracted Surname: {surname}")
    if special_number:
        print(f"Extracted Special Infomration: {special_number}")

    update_database(image_path, name, surname)