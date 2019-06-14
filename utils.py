import torch, glob, os
import numpy as np

def load_last(model, ckpt_root):
    pattern = os.path.join(ckpt_root, r'????.pkl')
    ckpts = glob.glob(pattern)
    if len(ckpts) == 0:
        return model, 0, []
    ckpts.sort()
    last_ckpt = ckpts[-1]
    state_dict = torch.load(last_ckpt, 
        map_location=lambda storage, location: storage.cuda() if torch.cuda.is_available() else storage)
    model.load_state_dict(state_dict)
    print(f'loaded {last_ckpt}')

    # get last epoch
    last_epoch = os.path.basename(last_ckpt).split('.')[0]
    last_epoch = int(last_epoch)

    # get acc hist
    acc_hist_path = os.path.join(ckpt_root, 'acc_hist.npy')
    if os.path.isfile(acc_hist_path):
        acc_hist = np.load(acc_hist_path).tolist()
    else:
        acc_hist = []
    return model, last_epoch, acc_hist

def load_best(model, ckpt_root):
    pattern = os.path.join(ckpt_root, r'????.pkl')
    ckpts = glob.glob(pattern)
    if len(ckpts) == 0:
        raise Exception('unable to find any weight file')
    ckpts.sort()

    # get acc hist
    acc_hist_path = os.path.join(ckpt_root, 'acc_hist.npy')
    if not os.path.isfile(acc_hist_path):
        raise Exception(f'unable to find {acc_hist_path}')
    acc_hist = np.load(acc_hist_path)

    # load best weights
    best_idx = np.argmax(acc_hist)
    best_ckpt = ckpts[best_idx]

    state_dict = torch.load(best_ckpt, 
        map_location=lambda storage, location: storage.cuda() if torch.cuda.is_available() else storage)
    model.load_state_dict(state_dict)
    print(f'loaded {best_ckpt}')

    return model