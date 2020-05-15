import cv2
import pandas as pd 
import numpy as np


eye_cascade = cv2.CascadeClassifier("frontalEyes35x16.xml")

nose_cascade = cv2.CascadeClassifier("Nose18x15.xml")

glasses = cv2.imread("glasses.png",cv2.IMREAD_UNCHANGED)
glasses = cv2.cvtColor(glasses,cv2.COLOR_BGRA2RGBA)

mustache = cv2.imread("mustache.png",-1)
mustache = cv2.cvtColor(mustache,cv2.COLOR_BGRA2RGBA)

img = cv2.imread("Before.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

eyes = eye_cascade.detectMultiScale(img,1.1,5)

x,y,w,h = eyes[0]
glasses = cv2.resize(glasses,(w,h))
for i in range(glasses.shape[0]):
    for j in range(glasses.shape[1]):
        if glasses[i,j,3] > 0:
            img[y+i,x+j,:] = glasses[i,j,:-1]

mst = nose_cascade.detectMultiScale(img,1.5,5)

x,y,w,h = mst[0]
mustache = cv2.resize(mustache,(w,h-10))
for i in range(mustache.shape[0]):
    for j in range(mustache.shape[1]):
        if mustache[i,j,3] > 0:
            img[y + int(h/2.0) + i,x+j,:] = mustache[i,j,:-1]

img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imwrite("img.jpg",img)
img = img.reshape((-1,3))
cv2.imshow("image",img)

df = pd.DataFrame(img, columns=["Channel 1","Channel 2", "Channel 3"]).to_csv("new_pred.csv", index=False)

cv2.waitKey(0)