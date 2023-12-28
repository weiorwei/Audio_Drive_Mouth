import os

root = "/opt/cjw_data/green_data/green_data_batch_3/"
p = os.listdir(root)
paths = []
for i in p:
    paths.append(os.path.join(root, i))
i = 0
for path in paths:
    video_format = os.path.basename(path).split(".")[-1]
    os.rename(path, os.path.join(root, "batch_3_{}".format(i) + "." + video_format))
    i = i + 1
