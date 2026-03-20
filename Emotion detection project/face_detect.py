import cv2

face = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
