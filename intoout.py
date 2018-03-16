import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
import face_recognition
from sklearn.svm import SVC
from PIL import Image, ImageDraw

v1 = cv2.VideoCapture("Young_blonde_woman_6.mp4")
frame_width = int(v1.get(3))
frame_height = int(v1.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

face_locations = []
face_encodings = []
face_emotions = []

process_this_frame = True

emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat")  # Use this to draw landmarks on detected face
clf = SVC(kernel='linear', probability=True,
          tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {}


def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset/%s/*" % emotion)
    random.shuffle(files)
    return files


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
        testvector = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


def make_sets():
    training_data = []
    training_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))
    return training_data, training_labels


training_data, training_labels = make_sets()
npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
npar_trainlabs = np.array(training_labels)
classifier = clf.fit(npar_train, training_labels)


# training part & pre define part finished here

# for prediction frame
def detect_emotion(image):
    testgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testclahe_image = clahe.apply(testgray)

    detections = detector(testclahe_image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(testclahe_image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

    result = clf.predict_proba(np.array(landmarks_vectorised).reshape(1, -1))
    return result


while True:
    ret, frame = v1.read()

    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)


    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_emotions = []
        face_emotions = detect_emotion(frame)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_emotions):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        # top *= 4
        # right *= 4
        # bottom *= 4
        # left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        # cv2.rectangle(frame, (0, 0), (80, 129), (0, 0, 255))
        font = cv2.FONT_HERSHEY_DUPLEX
        for i in range(0, 7):
            cv2.putText(frame, emotions[i], (10, i * 15 + 25), font, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, str(round(name[i], 4)), (100, i * 15 + 25), font, 0.5, (0, 0, 255), 1)

        print(name)

    out.write(frame)

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

v1.release()
out.release()
cv2.destroyAllWindows()
