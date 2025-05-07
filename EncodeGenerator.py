import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://faceattendancerealtime-2349f-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendancerealtime-2349f.firebasestorage.app"
})

# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)

imgList = []
studentIds = []

# Filter only image files (avoid .DS_Store or other non-image files)
valid_extensions = ('.png', '.jpg', '.jpeg')

for path in pathList:
    if path.lower().endswith(valid_extensions):  # Check if file is an image
        img = cv2.imread(os.path.join(folderPath, path))
        if img is not None:  # Ensure image is loaded properly
            imgList.append(img)
            studentIds.append(os.path.splitext(path)[0])  # Extract filename without extension
            fileName = f'{folderPath}/{path}'
            bucket = storage.bucket()
            blob = bucket.blob(fileName)
            blob.upload_from_filename(fileName)

print("Student IDs:", studentIds)

# Function to encode images
def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
            encodings = face_recognition.face_encodings(img)
            if encodings:  # Ensure at least one face is detected
                encodeList.append(encodings[0])
            else:
                print("Warning: No face detected in one of the images, skipping.")
        except Exception as e:
            print(f"Error processing image: {e}")
    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
