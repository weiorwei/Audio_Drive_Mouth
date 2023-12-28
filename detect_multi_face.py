import mediapipe as md
import numpy as np
import pylab
import imageio

import cv2
import os
import moviepy.editor as mp
from glob import glob
import mediapipe as md
import argparse


def cut_siginal(path):
    video_stream = cv2.VideoCapture(path)
    mp_face_detection = md.solutions.face_detection
    i=0
    while video_stream.isOpened():
        i+=1
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5,
                                             model_selection=1) as face_detection:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            H, W, _ = frame.shape
            if H < W:
                frame = frame[:H * 2 // 3, W // 4:W * 3 // 4, :]
            else:
                frame = frame[:H * 2 // 3, :, :]
            img = frame
            H, W, _ = img.shape
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections is not None:
                if len(results.detections) == 1:
                    continue
                else:
                    with open("Data_mark_two_people.txt", "a+") as f:
                        f.writelines(path + " frame-{}-have-two-people!!!".format(i) + "\n")
                    break
            else:
                with open("Data_mark_noface.txt", "a+") as f:
                    f.writelines(path + " frame-{}-no-face !!!".format(i) + "\n")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="video path",default=r"C:\Users\Administrator\Desktop\00013-10.mp4")
    args = parser.parse_args()
    cut_siginal(args.path)
