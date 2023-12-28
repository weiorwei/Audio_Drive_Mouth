import os
import glob
import multiprocessing
from tqdm import tqdm


pool = multiprocessing.Pool(64)
root = "/opt/cjw_data/green_data/green_data_slice_10/"
tem = os.listdir(root)
path = []
for i in tem:
    path.append(os.path.join(root, i))
# print(type(tem[0]))

for name in path:
    cmd = f"""python detect_multi_face.py --path {name} """
    pool.apply_async(os.system, args=(cmd,))

pool.close()
pool.join()