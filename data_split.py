import os, shutil, glob
import numpy as np


def run_split():
    DATA_ROOT = './data'
    GEN_DIR = os.path.join(DATA_ROOT, 'gen')
    TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
    VAL_DIR = os.path.join(DATA_ROOT, 'val')
    classes = os.listdir(GEN_DIR)
    val_ratio = 0.2
    for c in classes:
        rgb_files = glob.glob(os.path.join(GEN_DIR, c, '*.rgb.png'))
        rgb_files.sort()
        depth_files = glob.glob(os.path.join(GEN_DIR, c, '*.depth.exr'))
        depth_files.sort()
        assert len(rgb_files) == len(depth_files)
        
        indexes = np.arange(len(rgb_files))
        np.random.shuffle(indexes)
        val_len = int(len(indexes)*val_ratio)
        val_indexes = indexes[:val_len]
        train_indexes = indexes[val_len:]
        
        train_class_dir = os.path.join(TRAIN_DIR, c)
        print(f'copy {len(train_indexes)} to {train_class_dir}')
        os.makedirs(train_class_dir, exist_ok=True)
        for idx in train_indexes:
            shutil.copy(rgb_files[idx], train_class_dir)
            shutil.copy(depth_files[idx], train_class_dir)
            
        val_class_dir = os.path.join(VAL_DIR, c)
        print(f'copy {len(val_indexes)} to {val_class_dir}')
        os.makedirs(val_class_dir, exist_ok=True)
        for idx in val_indexes:
            shutil.copy(rgb_files[idx], val_class_dir)
            shutil.copy(depth_files[idx], val_class_dir)

if __name__ == "__main__":
    run_split()