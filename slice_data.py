import numpy as np
import pylab
import imageio

import cv2
import os
import moviepy.editor as mp
from glob import glob
import mediapipe as md
import matplotlib.pyplot as plt
from multiprocessing import Process
from demucs import pretrained
import torch
from demucs.apply import apply_model
import librosa
import soundfile as sf
import audio
import random
from hparams import hparams as hp
import torch.nn as nn
import datetime
import os, random, cv2, argparse
parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--drop", help="drop the front and the last 10s",required=True)

args = parser.parse_args()

def slice_video(video_path, save_path):
    slice_len = 5
    os.makedirs(save_path, exist_ok=True)
    for video in video_path:
        video_name = os.path.basename(video)[:-4]
        my_clip = mp.VideoFileClip(video)
        if args.drop:
            my_clip = my_clip.subclip(10, t_end=my_clip.duration - 10)
        if my_clip.duration > 12:
            for i in range(int(my_clip.duration // slice_len)):
                if i < (my_clip.duration // slice_len) - 1:
                    new_clip = my_clip.subclip(i * slice_len, (i + 1) * slice_len)
                    new_clip.write_videofile(
                        os.path.join(save_path, video_name + "-" + str(i) + ".mp4"), threads=4)
                else:
                    new_clip = my_clip.subclip(i * slice_len, my_clip.duration)
                    new_clip.write_videofile(
                        os.path.join(save_path, video_name + "-" + str(i) + ".mp4"),
                        threads=4)


def shoot_img(video_path, save_path):
    for video in video_path:
        count = 0
        video_cap = cv2.VideoCapture(video)
        print("Open Video ", video)
        while video_cap.isOpened():
            still_open, frame = video_cap.read()
            if not still_open:
                video_cap.release()
                break
            if count % 500 == 0:
                cv2.imwrite(os.path.join(save_path, os.path.basename(video)[:-4] + str(count) + ".png"), frame)
            count += 1
        print("Close Video ", video)


if __name__ == '__main__':
    root = r"/opt/cjw_data/green_data/pure_green/"
    video = os.listdir(root)
    video_path = []
    for v in video:
        video_path.append(os.path.join(root, v))
    save_path = r"/opt/cjw_data/green_data/green_data_slice_10/"
    slice_video(video_path, save_path)
    # shoot_img(video_path, save_path)
