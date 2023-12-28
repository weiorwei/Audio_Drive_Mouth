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

parser.add_argument("--data_train_txt", default="train.txt", help="Root folder of the preprocessed LRS2 dataset")
parser.add_argument("--data_val_txt", default="val.txt", help="Root folder of the preprocessed LRS2 dataset")
parser.add_argument('--checkpoint_dir', default="checkpoints_lr5e-5_beta1_0.5",
                    help='Save checkpoints to this directory', type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
parser.add_argument('--save_log_name', help='Resume quality disc from this checkpoint', required=True, type=str)
parser.add_argument('--True_ratio', help='Resume quality disc from this checkpoint', default=0.5, type=float)
parser.add_argument('--staic_true_ratio', help='Is every 10000*num step change True_ratio', default=True, type=bool)
parser.add_argument('--num', help='Is every 10000*num step change True_ratio', default=1, type=int)
parser.add_argument('--reset_optimizer', help='Is every 10000*num step change True_ratio', default=False, type=bool)
parser.add_argument('--pic_format', help='PNG / JPG /...', type=str, required=True)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

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


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    flag = 0
    resumed_step = global_step
    if local_rank == 0:
        with open(args.save_log_name, "a+") as f:
            f.writelines(
                str(datetime.datetime.now()) + "-Train Data-" + args.data_train_txt + "-" + "True Ratio-" + str(
                    args.True_ratio) + "-hparams.syncnet_batch_size :" + str(hparams.syncnet_batch_size)
                + "-hparams.syncnet_lr :" + str(hparams.syncnet_lr)
                + "-hparams.syncnet_betas :" + str(hparams.syncnet_betas)
                + " Begin Train!!! \n")
    print("Begin")
    while global_epoch < nepochs:

        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))

        for step, (x, mel, y) in prog_bar:

            model.train()
            optimizer.zero_grad()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step == 1 or global_step % checkpoint_interval == 0:
                if world_rank == 0:
                    if local_rank == 0:
                        with open(args.save_log_name, "a+") as f:
                            f.writelines(
                                str(datetime.datetime.now()) + "True Ratio-" + str(
                                    args.True_ratio) + "-Step:" + str(global_step) + '-Train Loss: {}'.format(
                                    running_loss / (step + 1)) + "\n")
                    save_checkpoint(
                        model, optimizer, global_step, checkpoint_dir, global_epoch)

            if global_step % hparams.syncnet_eval_interval == 0:
                with torch.no_grad():
                    flag = eval_model(test_data_loader, global_step, device, model, checkpoint_dir, flag)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, flag):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps:
                break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)
        with open(args.save_log_name, "a+") as f:
            f.writelines(
                str(datetime.datetime.now()) + "device_id: " + str(local_rank) + "-Step:" + str(
                    global_step) + '-Eval Loss: {}'.format(
                    averaged_loss) + "\n")

        flag = flag + 1
        if args.staic_true_ratio is False:
            if args.num == flag:
                if args.True_ratio == 0.5:
                    return
                elif args.True_ratio < 0.55 and args.True_ratio > 0.45:
                    args.True_ratio = 0.5
                elif args.True_ratio > 0.5:
                    args.True_ratio = args.True_ratio - 0.05
                else:
                    args.True_ratio = args.True_ratio + 0.05

        return flag


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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


if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_rank = torch.distributed.get_rank()

    # Dataset and Dataloader setup
    train_dataset = Dataset(args.data_train_txt)
    test_dataset = Dataset(args.data_val_txt)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, sampler=train_sampler,
        num_workers=hparams.num_workers, pin_memory=True,drop_last=True)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size, sampler=test_sampler,
        num_workers=hparams.num_workers, pin_memory=True,drop_last=True)

    # device_id = rank % torch.cuda.device_count()
    device = torch.device("cuda", local_rank)
    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr, betas=hparams.syncnet_betas)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=args.reset_optimizer)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model, device_ids=[device], output_device=local_rank)
    # model = DDP(model, broadcast_buffers=False, find_unused_parameters=True)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)
