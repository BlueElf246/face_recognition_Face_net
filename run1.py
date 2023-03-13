import cv2
from run import *
import glob
import pickle
model = InceptionResNetV1(
        input_shape=(None, None, 3),
        classes=128,
    )
model.load_weights('facenet_keras_weights.h5')
face_detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image_size=160
data= pickle.load(open('data_face.pkl', 'rb'))
def run():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,1.1, 4 )
        for (x,y, w, h) in faces:
            cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
            face= frame[y+2:y+h-2, x+2:x+w-2]
            area= w*h
            if area >11000:
                name, score = calc_dist(model=model,data=data,new_img=face)
                if score > 0.8:
                    cv2.putText(frame, 'unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                else:
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame.release()
run()