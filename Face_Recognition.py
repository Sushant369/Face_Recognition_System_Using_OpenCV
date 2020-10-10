import cv2
import numpy as np
import os
import time

global id1
def faceRec():


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('New1.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    id = 0

    c=True
    names = ['None', 'sushant shelar','rohan sukhadare', 'abhed mhatre', 'shaileshkumar yadav','parallel sushant'] 

    cam = cv2.VideoCapture(0)
    cam.set(3, 800) # set video widht
    cam.set(4, 800) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while c==True:
        ret, img =cam.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            
            if (confidence < 80 and confidence > 0):
                id1 = names[id]
                # print(confidence)
                # print(id)
                # return id1
                # c=False
                # break

            else:
                id1 = " "
    #             confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id1), (x+5,y-5), font, 1, (255,255,255), 2)
    #         cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            
            break
    # # Do a bit of cleanup

    cam.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":

    faceRec()
