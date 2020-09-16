#import CV2
import cv2

#Train the code
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detectfaces in

#img = cv2.imread('pic1.jpg')
img = cv2.imread('pic2.jpg')


# Must Convert to grey scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0 , 255), 3)


#Name For The Python Shell
cv2 .imshow('Daiwik Face Detector' , img)

#Runs The Python Shell For some time until Enter Key Is pressed
cv2.waitKey()

print("Code Complete!")
