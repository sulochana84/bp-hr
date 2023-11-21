""" Face detection from a video"""
import cv2
import os
import numpy as np
import glob
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
from optimalsize_face import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
def face_bbox(cap,x3,y3,x4,y4):
    """ this function """
    frame_num = 0
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        if ret == False:
            break
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 4)
        face_region = frame[y3: y4,x3: x4]
        cv2.imwrite('%s/%s.png' % ("outframe", frame_num ), face_region)
        cv2.imshow('Face Detection - OpenCV', frame)
        height = face_region .shape[0]
        width = face_region .shape[1]
        frame_num += 1
        if cv2.waitKey(10) == 27:
             break  # Wait Esc key to end prog
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def face_boundingbox_size(cap):
    """ this function is to calculate the face bounding box of
    the first frame and return coordianted"""
    while (True):
        ret, frame = cap.read()
        print(ret)
        frame = cv2.flip(frame, 0)
        if ret == False:
            break
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                #if x & y >0 :
                return x,y,x2,y2
            break
        else:
             print("Error Occured")


"""if __name__ == '__main__':

    cap1 =("/Users/mohan/Sulochana/KlarityAI/IMG_1424.MOV")
    ffmpeg_extract_subclip(cap1, 5, 15, targetname="clipped_video/test.MOV")
    cap1 = cv2.VideoCapture( "clipped_video/test.MOV")
    x,y,x2,y2  =  face_boundingbox_size( cap1)
    print(x)
    print(y)
    face_bbox(cap1, x, y, x2, y2)"""





