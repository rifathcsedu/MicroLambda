import json
import face_recognition
import time
import base64
import numpy as np
import pickle

#biden encodings
def BidenEncoding(data):
    return face_recognition.face_encodings(pickle.loads(data[0]))[0]

#compare between 2 images
def ImageCompare(first, second):
    result = face_recognition.compare_faces([first], second)
    return result
    
