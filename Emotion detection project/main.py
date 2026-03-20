import cv2
import numpy as np
from keras.models import load_model
import sqlite3
from datetime import datetime

# Load emotion model (no compile)
emotion_model = load_model("models/emotion_model.hdf5", compile=False)

# Load face detector
face = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h,x:x+w]
        roi = cv2.resize(roi,(64,64))
        roi = roi/255.0
        roi = np.reshape(roi,(1,64,64,1))


        pred = emotion_model.predict(roi)
        emotion = labels[np.argmax(pred)]

        gender = "Male"   # temporary (we add real gender later)

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        conn = sqlite3.connect("database/emotion.db")
        c = conn.cursor()
        c.execute("INSERT INTO logs(emotion,gender,date,time) VALUES(?,?,?,?)",
                  (emotion,gender,date,time))
        conn.commit()
        conn.close()

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,emotion,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
