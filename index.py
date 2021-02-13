import cv2
import numpy
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('images/obama.jpg')

# Get grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 7)

resized = []

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),3)
    resized_img = img[y:y+h, x:x+w]
    resized.append(resized_img)


if len(resized) == 0:
    print("No faces detected...")
    quit()

gender_net = cv2.dnn.readNetFromCaffe('age_gender_net/deploy_gender.prototxt', 'age_gender_net/gender_net.caffemodel')

age_net = cv2.dnn.readNetFromCaffe('age_gender_net/deploy_age.prototxt', 'age_gender_net/age_net.caffemodel')

gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

def predictAgeGender(faces):
    blob = cv2.dnn.blobFromImages(faces, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    genders = gender_net.forward()

    age_net.setInput(blob)
    ages = age_net.forward()
    print(genders)
    print()
    print(ages)
    print()
    labels = ['{}, {}'.format(gender_list[gender.argmax()], age_list[age.argmax()]) for (gender, age) in zip(genders, ages)]
    return labels

labels = predictAgeGender(resized)

font = cv2.FONT_HERSHEY_SIMPLEX

for face in faces:
    for label in labels:
        cv2.putText(img, label, (face[0], face[1] + 30), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

cv2.imshow('image', img)
# for pic in resized:
#     counter = 0
#     cv2.imshow('img'+ str(counter), pic)
#     counter += 1
cv2.waitKey(0)
cv2.destroyAllWindows()
