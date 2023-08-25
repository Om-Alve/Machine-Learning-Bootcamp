import cv2
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,BatchNormalization,Dropout
from keras.optimizers import Adam,SGD
from keras.applications.vgg16 import VGG16

model = keras.models.load_model('saved_model/')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
while True:
    ret,frame = cam.read()
    if ret == False:
        break
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        face = frame[y:y+h,x:x+w]
        img = cv2.resize(frame,(150,150))
        img =img.reshape(1,150,150,3)
        img = tf.cast(img,tf.float32)
        img = img / 255
        pred = "Mask" if model.predict(img) <= 0.5 else "No Mask"
        cv2.putText(frame,str(pred),(x,y-10),cv2.FONT_HERSHEY_COMPLEX,2,(255, 0, 0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0))
    frame=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Webcam",frame)
    
    
    
    print(pred)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
cv2.destroyAllWindows() 