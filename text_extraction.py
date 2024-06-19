import difflib
from keras_ocr.pipeline import Pipeline
import pandas as pd
import cv2
import re
import warnings

warnings.filterwarnings("ignore")

common_names = pd.read_csv("names.csv")['imie'].tolist()[:100]
common_surnames = pd.read_csv("polish_surnames.csv")['surname'].tolist()[:500]

pipeline = Pipeline()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(binary)
    processed_image = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    return processed_image

def recognize_text(image):
    prediction_groups = pipeline.recognize([image])
    predictions = prediction_groups[0]
    words = [word for word, _ in predictions]
    if words:
        print(f"All extracted words: {words}")
    return words

def find_closest_match(word, word_set):
    word_upper = word.upper()
    matches = difflib.get_close_matches(word_upper, word_set, n=1, cutoff=0.8)
    if matches:
        return matches[0]
    return word_upper

def extract_information(words):
    name = None
    surname = None
    special_number = None

    # Exact matches
    for word in words:
        word_upper = word.upper()
        if word_upper in common_names:
            name = word_upper
        elif word_upper in common_surnames:
            surname = word_upper
        elif re.match(r'\b[0-9]{4,10}\b', word):
            special_number = word

    # Closest matches (only if not found in first pass)
    if not name:
        for word in words:
            potential_name = find_closest_match(word, common_names)
            if potential_name != word.upper():
                name = potential_name
                break

    if not surname:
        for word in words:
            potential_surname = find_closest_match(word, common_surnames)
            if potential_surname != word.upper():
                surname = potential_surname
                break

    return name, surname, special_number