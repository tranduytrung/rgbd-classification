import torch, torch.utils.data
import torchvision
import os, glob
import loaders

class RGBDDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, rgb_extension='png', d_extension='exr', 
        loader=loaders.RGBDLoader(), transform=None):
        super(RGBDDataset, self).__init__()
        self.root = data_root
        self.loader = loader
        self.transform = transform
        self.classes = next(os.walk(data_root))[1]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_root, class_name)
            rgb_files = glob.glob(os.path.join(class_path, '*' + rgb_extension))
            d_files = glob.glob(os.path.join(class_path, '*' + d_extension))
            # make sure it match each other
            rgb_files.sort()
            d_files.sort()
            rgb_len = len(rgb_files)
            d_len = len(d_files)
            assert rgb_len == d_len, f'RGB files is not matched with Depth files ({rgb_len} != {d_len})'

            for jdx in range(rgb_len):
                rgb_file = rgb_files[jdx]
                # may get dir or link
                if not os.path.isfile(rgb_file):
                    continue

                d_file = d_files[jdx]
                self.samples.append([idx, rgb_file, d_file])

    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        cid, rgb_path, d_path = sample = self.samples[index]
        data = self.loader(rgb_path, d_path)

        if self.transform is not None:
            data = self.d_transform(data)

        return data, torch.tensor(cid, dtype=torch.long)