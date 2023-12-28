# -*- coding: utf-8 -*-
import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import listdir, path
from joblib import Parallel, delayed
# if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
#     raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
#                             before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
# import audio
from multiprocessing import Process
import mediapipe as md

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", default=r"C:\Users\Administrator\Desktop\green_data",
                    help="Root folder of the LRS2 dataset")
parser.add_argument("--video_path_list", default=None,
                    help="Root folder of the LRS2 dataset")
parser.add_argument("--preprocessed_root", default=r'C:\Users\Administrator\Desktop\green_processed',
                    help="Root folder of the preprocessed dataset")
parser.add_argument("--del_time", default=0, #required=True,
                    help="Cut video front and last time", type=int)
parser.add_argument("--Cut_Pic", default=True, #required=True,
                    help="If video quilty is too high,detect the whole pic will be slow,cut the mid of pic to detect the people face(known that people face will be in the middle of the video)",
                    type=bool)
args = parser.parse_args()

template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'


def process_video_file(files, args):
    mp_face_detection = md.solutions.face_detection
    length = len(files)
    del_time = args.del_time  # 删除前后 10秒的视频
    for vfile in files:
        video_stream = cv2.VideoCapture(vfile)
        frames = []
        while 1:
            still_reading, frame = video_stream.read()

            if not still_reading:
                video_stream.release()
                break
            H, W, _ = frame.shape
            if args.Cut_Pic:
                if H < W:
                    frame = frame[:H * 2 // 3, W // 4:W * 3 // 4, :]
                else:
                    frame = frame[:H * 2 // 3, :, :]
            frames.append(frame)

        video_length = len(frames)
        frames = frames[25 * del_time:video_length - args.del_time * 25]
        vidname = os.path.basename(vfile)[:-4]
        dirname = vfile.split('/')[-2]

        fulldir = path.join(args.preprocessed_root, dirname, vidname)
        os.makedirs(fulldir, exist_ok=True)

        i = -1
        for fb in frames:
            H, W, _ = fb.shape
            with mp_face_detection.FaceDetection(min_detection_confidence=0.5,
                                                 model_selection=1) as face_detection:
                results = face_detection.process(cv2.cvtColor(np.asarray(fb), cv2.COLOR_BGR2RGB))

                i = i + 1
                if results.detections and len(results.detections) == 1:
                    bounding_box = results.detections[0].location_data.relative_bounding_box
                    x1 = int(max(0, bounding_box.xmin * W))
                    y1 = int(max(0, (bounding_box.ymin) * H))
                    x2 = int(min(W, (bounding_box.xmin + bounding_box.width) * W))
                    y2 = int(min(H, ((bounding_box.ymin + bounding_box.height)) * H))

                    y1 -= int((y2 - y1) * 0.2)
                    y1 = max(0, y1)

                    cv2.imwrite(path.join(fulldir, '{}.png'.format(i)), fb[y1:y2, x1:x2])


def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args.preprocessed_root, dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')

    command = template.format(vfile, wavpath)
    subprocess.call(command, shell=True)


# def multi_process(filelist, args):
#     delay_call = [delayed(process_video_file)(file, args) for file in filelist]
#     p=Parallel(n_jobs=-1,backend="threading")
#     p([delay_call])

def main(args):
    if args.video_path_list is not None:
        filelist = [args.video_path_list]
    else:
        filelist = []
        video = os.listdir(args.data_root)
        for v in video:
            filelist.append(os.path.join(args.data_root, v))

    with open("count.txt", "a+") as f:
        f.writelines(filelist[0] + '\n')
    # multi_process(filelist, args)
    process_video_file(filelist, args)

    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except:
            traceback.print_exc()
            continue


if __name__ == '__main__':
    main(args)
