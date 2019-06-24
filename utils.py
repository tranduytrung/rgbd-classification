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

def bytescaling(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has 
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)