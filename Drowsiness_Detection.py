from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
from playsound import playsound 
from twilio.rest import Client 
from geopy.geocoders import Nominatim  
import requests
import time

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_current_location():
    latitude = 16.6000 
    longitude = 79.7333  
    return latitude, longitude


def upload_to_imgur(image_path):
    client_id = '0da250f4edece3d'  
    headers = {"Authorization": f"Client-ID {client_id}"}
    with open(image_path, "rb") as image_file:
        response = requests.post("https://api.imgur.com/3/image", headers=headers, files={"image": image_file})
    data = response.json()
    if response.status_code == 200:
        return data['data']['link']  
    else:
        print("Failed to upload image to Imgur:", data)
        return None

def send_emergency_alert(posture_image_url):
    account_sid = 'ACb7c4c60c340d96a1aa084c0e21f24fcc'
    auth_token = '70fbc118abadc57f1ffdc90e66802daf'
    client = Client(account_sid, auth_token)

    latitude, longitude = get_current_location()
    
    geolocator = Nominatim(user_agent="DrowsinessDetectionApp")  
    try:
        location = geolocator.reverse((latitude, longitude))
        location_description = location.address if location else "Unknown Location"
    except Exception as e:
        location_description = "Location could not be determined"
        print(f"Error retrieving location: {e}")

    message = client.messages.create(
        body=f"Emergency Alert: Driver is unresponsive. Current location: {location_description} (Lat: {latitude}, Lon: {longitude}). Last posture image: {posture_image_url}",
        from_='+16319252259',  
        to='+919866834960'      
    )
    print("Emergency alert sent to contact with location and posture image.")

thresh = 0.25
frame_check = 25
flag = 0
alarm_playing = False

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)

detection_start_time = None
while True:
    ret, frame = cap.read()
    if not ret:
        break  

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            print(f"Flag count: {flag}")

            if flag >= frame_check and detection_start_time is None:
                detection_start_time = time.time() 

                posture_image_path = 'last_posture.png'
                cv2.imwrite(posture_image_path, frame)

                if not alarm_playing:
                    playsound('alarm.wav', block=False)  
                    alarm_playing = True

                if flag >= 100:
                    posture_image_url = upload_to_imgur(posture_image_path)
                    if posture_image_url:
                        send_emergency_alert(posture_image_url)
                    flag = 0 
        else:
            flag = 0
            alarm_playing = False
            detection_start_time = None  

    if detection_start_time and (time.time() - detection_start_time) <= 15:
        cv2.putText(frame, "!!! DROWSINESS DETECTED !!!", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()