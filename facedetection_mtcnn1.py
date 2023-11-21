""" Face detection from a video"""
import cv2
import os
import numpy as np
import glob
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
from optimalsize_face import *
def face_bbox(cap,x3,y3,x4,y4):
    frame_num = 0
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        print(ret)
        if ret == False:
            break
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 4)
        face_region = frame[y3: y4,x3: x4]
        cv2.imwrite('%s/%s.png' % ("/Users/mohan/PycharmProjects/KlarityAII/outframe", frame_num ), face_region)
        cv2.imshow('Face Detection - OpenCV', frame)
        height = face_region .shape[0]
        width = face_region .shape[1]
        print(height)
        print(width)
        frame_num += 1
        if cv2.waitKey(10) == 27:
             break  # Wait Esc key to end prog
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def face_size(cap):
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        print(ret)
        if ret == False:
            break
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
    return x,y,x2,y2

def face_size_frame(cap):
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        print(ret)
        if ret == False:
            break
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                if x & y >0 :
                  return x,y,x2,y2
                  print(x)
                  break

if __name__ == '__main__':
    cap1 = cv2.VideoCapture("/Users/mohan/Sulochana/KlarityAI/IMG_1424.MOV")
    x,y,x2,y2  =   face_size_frame( cap1)
    print(x)
    print(y)
    face_bbox(cap1, x, y, x2, y2)





