import os
import glob
import  multiprocessing 
from tqdm import tqdm

  
pool=multiprocessing.Pool(64)

root="/opt/cjw_data/green_data/green_data_slice_batch_3_cleaned/"       
tem=os.listdir(root)
path=[]
for i in tem:
    path.append(os.path.join(root,i))
#print(type(tem[0]))
    
for name in path:
    cmd=f"""python preprocess_mine.py --video_path_list {name} --preprocessed_root="/opt/cjw_data/green_data/green_data_process/" """
    pool.apply_async(os.system,args=(cmd,))

pool.close()
pool.join()