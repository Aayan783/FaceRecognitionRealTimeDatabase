import cv2
import os
import pickle
import numpy as np
import cvzone
import face_recognition
import firebase_admin
from firebase_admin import credentials, db, storage
from datetime import datetime

# Firebase initialization
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-2349f-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-2349f.appspot.com"
})
bucket = storage.bucket()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Background and mode images
imgBackground = cv2.imread('Resources/background.png')
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load Encoded File
print("Loading Encode File...")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Variables
modeType = 0
counter = 0
id = -1
imgStudent = []
recognized = False

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    recognized = False

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

            id = studentIds[matchIndex]
            recognized = True

            if counter == 0:
                counter = 1
                modeType = 1

    if recognized:
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[0]
    else:
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]

    if counter != 0:
        if counter == 1:
            studentInfo = db.reference(f'Students/{id}').get()
            print("Student Data:", studentInfo)

            blob = None
            for ext in ['.png', '.jpg', '.jpeg']:
                try:
                    blob = bucket.get_blob(f'Images/{id}{ext}')
                    if blob:
                        break
                except:
                    continue

            if blob:
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
            else:
                print(f"[Warning] Student image not found in Firebase Storage for ID: {id}")
                imgStudent = np.zeros((216, 216, 3), dtype=np.uint8)

            # Mark attendance once per session
            last_attendance_time = studentInfo.get("last_attendance_time", "")
            try:
                last_time = datetime.strptime(last_attendance_time, "%Y-%m-%d %H:%M:%S")
            except:
                last_time = datetime(2000, 1, 1)

            if (datetime.now() - last_time).total_seconds() > 30:
                ref = db.reference(f'Students/{id}')
                studentInfo['total_attendance'] += 1
                studentInfo['last_attendance_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ref.set(studentInfo)

        # Display student info
        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(id), (1006, 493),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

        (w, _), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        offset = (414 - w) // 2
        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent
        counter += 1

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
