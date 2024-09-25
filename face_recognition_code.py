from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__, template_folder='templates')

# Path where attendance images are stored
path = 'imagesAttendance'
images = []
classnames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])


# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# Function to mark attendance in CSV
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDatalist = f.readlines()
        namelist = [line.split(',')[0] for line in myDatalist]

        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


# Find encodings for known faces
encodeListKnown = findEncodings(images)
print('Encoding complete')


# Route for the homepage
@app.route('/')
def home():
    return render_template('home.html')


# Route for face recognition and attendance marking
@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(BytesIO(img_data))
    img = np.array(img)

    # Resize for faster processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            markAttendance(name)
            return jsonify({"message": f'Attendance marked for {name}'})

    return jsonify({"message": "No match found"})


if __name__ == '__main__':
    app.run(debug=True)
