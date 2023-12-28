import moviepy.editor as mp
import os
from glob import glob
import shutil


def calculate_data_sync(root):
    data_name = os.listdir(root)
    for data in data_name:
        path = os.path.join(root, data)
        pic = glob(os.path.join(path, "*.jpg"))
        my_clip = mp.AudioFileClip(os.path.join(path, "audio.wav"))
        fps = len(pic) / my_clip.reader.duration
        with open("caculate_data_sync.txt", "a+") as f:
            f.writelines(path + ": " + str(fps) + "\n")


def pack_data(root, save_bil, save_xhs):
    data_names = os.listdir(root)
    for name in data_names:
        split_name = name.split("_")
        if split_name[1] == "bili":
            shutil.copytree(os.path.join(root, name),
                            os.path.join(save_bil, split_name[1] + "_" + split_name[2], split_name[3]))
        elif split_name[0] == "xiaohongshu":
            if len(split_name) == 3:
                shutil.copytree(os.path.join(root, name),
                                os.path.join(save_xhs, split_name[0] + "_" + split_name[1], split_name[2]))
            else:
                shutil.copytree(os.path.join(root, name),
                                os.path.join(save_xhs, split_name[0] + "_" + split_name[1] + "_" + split_name[2],
                                             split_name[3]))


def rename_vocal(root):
    data_type = os.listdir(root)
    for data in data_type:
        videos_path = os.listdir(os.path.join(root, data))
        for video in videos_path:
            if os.path.exists(os.path.join(root, data, video, "audio.wav")) and os.path.exists(
                    os.path.join(root, data, video, "vocals.wav")):
                os.remove(os.path.join(root, data, video, "audio.wav"))
                os.rename(os.path.join(root, data, video, "vocals.wav"),
                          os.path.join(root, data, video, "audio.wav"))


def rename_dir(root):
    dirs = os.listdir(root)
    i = 0
    for d in dirs:
        os.rename(os.path.join(root, d), os.path.join(root, "%05d" % i))
        i = i + 1


if __name__ == "__main__":
    # root = "/opt/cjw_data/xiao_hong_book_data/bil_xhs_process/"
    # calculate_data_sync(root)

    # root = "/opt/cjw_data/xiao_hong_book_data/xhs_data/"
    # save_bil = "/opt/cjw_data/xiao_hong_book_data/bili_data_pack/"
    # save_xhs = "/opt/cjw_data/xiao_hong_book_data/xhs_data_pack/"
    # pack_data(root, save_bil, save_xhs)

    root = "/opt/cjw_data/data_mine/"
    rename_vocal(root)

    # root = "/opt/cjw_data/LRS2/mvlrs_v1/main/"
    # rename_dir(root)
