import datetime
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color_hq as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
from torch.nn.parallel import DistributedDataParallel as DDP
import os, random, cv2, argparse
from hparams import hparams, get_image_list
import torch.distributed as dist
import datetime

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", default="filelists", help="Root folder of the preprocessed LRS2 dataset")
parser.add_argument("--data_train_txt", default="train_all_data.txt",
                    help="Root folder of the preprocessed LRS2 dataset")
parser.add_argument("--data_val_txt", default="val_all_data.txt", help="Root folder of the preprocessed LRS2 dataset")
parser.add_argument('--checkpoint_dir', default="checkpoints", help='Save checkpoints to this directory', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint',
                    default="checkpoints/lipsync_expert.pth", required=True,
                    type=str)
parser.add_argument('--fine_tune', help='', default=False, type=bool)
parser.add_argument('--True_ratio', help='', default=0.5, type=float)
parser.add_argument('--pic_format', help='PNG / JPG /...', type=str, required=True)
args = parser.parse_args()

syncnet_T = 5
syncnet_mel_step_size = 16


class Dataset(Dataset):
    def __init__(self, data_txt):
        self.count = 0
        self.all_videos = []
        with open(data_txt) as f:
            for lines in f.readlines():
                lines = lines.strip()
                self.all_videos.append(lines)

    def get_frame_id(self, frame):
        if "error" in basename(frame).split('.')[0]:
            return None
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        if start_id is None:
            return None
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.'.format(frame_id)+args.pic_format)
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        if start_frame_num is None:
            return None
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.'+args.pic_format)))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            x = random.randint(1, 20)
            # if random.choice([True, False]):
            #     y = torch.ones(1).float()
            #     chosen = img_name
            # else:
            #     y = torch.zeros(1).float()
            #     chosen = wrong_img_name

            if x <= 20 * args.True_ratio:
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                try:
                    wavpath = join(vidname, "audio.wav")
                except:
                    wavpath = join(vidname, "vocals.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)
            if mel is None:
                continue
            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1] // 2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            break

        return x, mel, y


def cal_accuracy(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    d = torch.round(d)
    acc = y.shape[0] - (torch.sum(torch.abs(d.unsqueeze(1) - y)))
    return acc


global_step = 0
global_epoch = 0


def eval(device, model, test_data_loader, optimizer, nepochs=None):
    global global_step, global_epoch
    resumed_step = global_step
    acc = 0
    total_item = 0
    with open("Eval_sync_log.txt", "a+") as f:
        f.writelines("Begin Eval on " + args.data_val_txt + "\n")
        f.writelines(str(datetime.datetime.now()) + args.checkpoint_path + "-True Ratio:" + str(
            args.True_ratio) + "\n")

    model.eval()
    while global_epoch < nepochs:

        running_loss = 0.
        prog_bar = tqdm(enumerate(test_data_loader))
        with torch.no_grad():
            for step, (x, mel, y) in prog_bar:
                model.train()

                # Transform data to CUDA device
                x = x.to(device)

                mel = mel.to(device)
                a, v = model(mel, x)
                y = y.to(device)

                acc = (acc + cal_accuracy(a, v, y))
                total_item = total_item + x.shape[0]

            with open("Eval_sync_log.txt", "a+") as f:
                f.writelines(str(datetime.datetime.now()) + "Total Item: " + str(total_item) + "accuracy:" + str(
                    (acc / total_item * 100).item()) + "%" + "\n")
            print("Total Item:", total_item)
            print("accuracy:", (acc / total_item * 100).item(), "%")

        global_epoch += 1


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        new_checkpoint = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(new_checkpoint)

    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model


use_cuda = torch.cuda.is_available()
# print('use_cuda: {}'.format(use_cuda))
if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup

    test_dataset = Dataset(args.data_val_txt)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)
    eval(device, model, test_data_loader, optimizer, nepochs=hparams.nepochs)
