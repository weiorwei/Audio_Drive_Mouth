import os
import random
import shutil
from glob import glob


def move_tiktok_data(root):
    video_type = os.listdir(root)
    for video in video_type:
        video_list = os.listdir(os.path.join(root, video))
        for v in video_list:
            shutil.copytree(os.path.join(root, video, v), "tiktok_out_data/")
            print("Move ", os.path.join(root, video, v), " to tiktok_out_data/")


def get_path_fine_tune_txt(data_root, split):
    videos_type = os.listdir(os.path.join(data_root, split))
    for videos in videos_type:
        video_path = os.listdir(os.path.join(data_root, split, videos))
        for path in video_path:
            slice_video = os.listdir(os.path.join(data_root, split, videos, path))
            for slice in slice_video:
                with open("fine_tune_" + split + ".txt", "a+") as f:
                    f.writelines(os.path.join(data_root, split, videos, path, slice) + "\n")


def get_not_out_path_fine_tune_txt(data_root):
    video_type_list = os.listdir(data_root)
    for video_type in video_type_list:
        videos = os.listdir(os.path.join(data_root, video_type))
        for video in videos:
            slices = os.listdir(os.path.join(data_root, video_type, video))
            for s in slices:
                flag = random.randint(1, 20)
                if flag != 1:
                    with open("train" + ".txt", "a+") as f:
                        f.writelines(os.path.join(data_root, video_type, video, s) + "\n")
                else:
                    with open("val" + ".txt", "a+") as f:
                        f.writelines(os.path.join(data_root, video_type, video, s) + "\n")


def get_path_txt(data_root):
    videos_type = os.listdir(data_root)
    for path in videos_type:
        flag = random.randint(1, 20)
        if flag != 1:
            with open("train_path/" + "train" + "_green" + ".txt", "a+") as f:
                f.writelines(os.path.join(data_root, path) + "\n")
        else:
            with open("train_path/" + "val" + "_green" + ".txt", "a+") as f:
                f.writelines(os.path.join(data_root, path) + "\n")


def get_mp4():
    filelist = glob(os.path.join("/opt/cjw_data/xiao_hong_book_data/bil_xhs_data/bil_xhs_data_slice/", '*/*.mp4'))
    with open("xhs_mp4" + ".txt", "a+") as f:
        for path in filelist:
            f.writelines(path + "\n")


def get_xhs_bil_path(root):
    videos = os.listdir(root)
    for video in videos:
        flag = random.randint(1, 10)
        if flag > 2:
            with open("train_path/" + "train" + ".txt", "a+") as f:
                f.writelines(os.path.join(root, video) + "\n")
        else:
            with open("train_path/" + "val" + ".txt", "a+") as f:
                f.writelines(os.path.join(root, video) + "\n")


def rename_data(path):
    count = 0
    for p in path:
        video_name = os.path.basename(p)
        root = os.path.dirname(p)
        format_p = video_name.split(".")[-1]
        os.rename(p, os.path.join(root, "batch_2_" + str(count) + "." + format_p))
        count += 1


if __name__ == "__main__":
    data_root = r"/opt/cjw_data/green_data/green_data_process/green_data_slice_batch_2_cleaned/"
    get_path_txt(data_root)

    #root = "/opt/hl_data/green_data/"
    #list_v = os.listdir(root)
    #path = []
    #for v in list_v:
    #    path.append(os.path.join(root, v))
    #rename_data(path)
