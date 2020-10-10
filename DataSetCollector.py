import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# noseCascade = cv2.CascadeClassifier('Nariz.xml')
# mouthCascade = cv2.CascadeClassifier('Mouth.xml')
cam = cv2.VideoCapture(0)
cam.set(3, 300) # set video widht
cam.set(4, 300) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

id=input("enter the user id: ")

sampleNum=0


while True:
    
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )


    # faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # eyes=eye_cascade.detectMultiScale(gray,1.1,12)
    # nose=noseCascade.detectMultiScale(gray,1.1,4)
    

    for (x,y,w,h) in faces:
        sampleNum=sampleNum+1

        cv2.imwrite("NewDB/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
        cv2.imshow('img',img)
        # cv2.waitKey(1)
        if sampleNum >=20:
            cam.release()
            cv2.destroyAllWindows()
            break

    # if (sampleNum>=10):
    #     for (x,y,w,h) in eyes:
    #         sampleNum=sampleNum+1
    #         cv2.imwrite("NewDB/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    #         cv2.waitKey(100)
    #         cv2.imshow('img',img)
    #         cv2.waitKey(1)
        # if (sampleNum>=20):
        #             for (x,y,w,h) in nose:
        #                 sampleNum=sampleNum+1
        #                 cv2.imwrite("nose/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        #                 cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        #                 cv2.waitKey(100)
        #                 cv2.imshow('img',img)
        #                 cv2.waitKey(1)
        #             if (sampleNum>=30):
        #                 break
                    
                   
       



    

