import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import os
from flask import Flask, Response
import datetime
import serial
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from telegram import Bot
import asyncio
import urllib.request
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials and configurations from .env
ESP32_IP = os.getenv('ESP32_IP')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
CAMERA_URL = os.getenv('CAMERA_URL')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

if not all([ESP32_IP, GMAIL_USER, GMAIL_PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CAMERA_URL, RECIPIENT_EMAIL]):
    raise ValueError("Some environment variables are missing. Please check your .env file.")

# Function to trigger the buzzer
def trigger_buzzer():
    response = requests.get(f"http://{ESP32_IP}/trigger_buzzer")
    if response.status_code == 200:
        print("Buzzer triggered!")

# Function to lock the door
def lock_door():
    response = requests.get(f"http://{ESP32_IP}/lock_door")
    if response.status_code == 200:
        print("Door locked!")

# Create an SMTP client
smtp_client = smtplib.SMTP("smtp.gmail.com", 587)
smtp_client.starttls()
smtp_client.login(GMAIL_USER, GMAIL_PASSWORD)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load known face encodings and names for face recognition
path = r'C:\Users\HP\Desktop\camvisiotech\test'  # Replace with your image directory
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Variables for "0" counting and last notification time
continuous_zeros = 0
last_notification_time = None

def generate_frames():
    global continuous_zeros, last_notification_time

    while True:
        try:
            # Capture the image from the ESP32-CAM's URL
            img_response = urllib.request.urlopen(CAMERA_URL)
            img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, -1)

            # Convert the frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(image_rgb)

                # Initialize flags
                recognized = False
                unknown = False

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                        # Crop and process the face
                        face_image = frame[y:y+h, x:x+w]
                        if face_image.shape[0] > 0 and face_image.shape[1] > 0:
                            imgS = cv2.resize(face_image, (0, 0), None, 0.25, 0.25)
                            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                            facesCurFrame = face_recognition.face_locations(imgS)
                            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

                            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                                if any(matches):
                                    recognized = True
                                    name = classNames[matches.index(True)]
                                    y1, x2, y2, x1 = faceLoc
                                    y1, x2, y2, x1 = y1 * 4 + y, x2 * 4 + x, y2 * 4 + y, x1 * 4 + x
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                                    cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                else:
                                    unknown = True
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Handle recognition status
            if recognized or not (recognized or unknown):
                continuous_zeros = 0
            else:
                continuous_zeros += 1

            if continuous_zeros >= 9:  # 3 seconds of "0"
                current_time = datetime.datetime.now()
                if last_notification_time is None or (current_time - last_notification_time).total_seconds() >= 15:
                    last_notification_time = current_time
                    trigger_buzzer()
                    lock_door()
                    send_notification("Motion Detected!", "An intruder was detected!")

            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error: {e}")

app = Flask(__name__)

@app.route('/')
def index():
    return "Livestreaming Face Recognition/Detection"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

async def send_telegram_message(text):
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

def send_notification(subject, body):
    send_email(subject, body)
    asyncio.run(send_telegram_message(f"{subject}\n{body}"))

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = GMAIL_USER
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    smtp_client.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
