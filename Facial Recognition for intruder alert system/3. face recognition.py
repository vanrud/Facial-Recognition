

import cv2
'''import numpy as np
import os '''


import requests



recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')   #load trained model
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# number of persons
id = 5


names = ['','Durnav','Sreejith', 'Darrshann', 'Bala', 'Mithun']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
# set video width
cam.set(3, 640)

# set video height
cam.set(4, 480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
count = 0

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        id , confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 90 ):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            id = "!!! user not recognized !!!"
            confidence = "  {0}%".format(round(confidence))
            count+=1
            if count == 70:
                print("!!!!  INTRUSION ALERT - - - - - -  UNAUTHORIZED PERSONNEL !!!!")
                response1=requests.post('https://maker.ifttt.com/trigger/INTRUSIONALERT/json/with/key/ITOZ4XvTNCw32dRjIyi0u')


                if response1.status_code == 200:
                    print("Successfully notified ")
                else:
                    print("request failed ")



        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
    
    cv2.imshow('camera', img)



    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
