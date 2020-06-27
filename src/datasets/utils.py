import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.utils.data as data
from .eurosat import EuroSAT
from .voc_seg_dataset import VOCSegDataset
import os
import six
from PIL import Image
import numpy as np
from torch.utils.model_zoo import tqdm

datasets = {
    'eurosat': EuroSAT,
    'voc': VOCSegDataset
}

def get_dataset(name, **kwargs):
    print(kwargs)
    if name.lower() not in datasets:
        raise ValueError("no dataset named {}, should be one of {}".format(name, ' '.join(datasets)))
    return datasets.get(name.lower())(**kwargs)

def calc_normalization(train_dl: torch.utils.data.DataLoader):
    "Calculate the mean and std of each channel on images from `train_dl`"
    mean = torch.zeros(3)
    m2 = torch.zeros(3)
    n = len(train_dl)
    for images, labels in tqdm(train_dl, "Compute normalization"):
        mean += images.mean([0, 2, 3]) / n
        m2 += (images ** 2).mean([0, 2, 3]) / n
    var = m2 - mean ** 2
    return mean, var.sqrt()

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def random_split(dataset, ratio=0.9, random_state=None):
    """ split dataset

    Args:
        dataset (Dataset): Dataset to be split
        ratio (float): Split rate of training and testing.
        random_state (int): The desired seed.

    Returns:

    """
    if random_state is not None:
        state = torch.random.get_rng_state()
        torch.random.manual_seed(random_state)
    n = int(len(dataset) * ratio)
    split = torch.utils.data.random_split(dataset, [n, len(dataset) - n])
    if random_state is not None:
        torch.random.set_rng_state(state)
    return split

def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # downloads file
    if os.path.isfile(fpath):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )


def tifffile_loader(path):
    # all the loader should be numpy ndarray [height, width, channels]
    # int16: (-32768 to 32767)
    import tifffile
    img = tifffile.imread(path)
    if img.dtype in [np.uint8, np.uint16, np.float]:
        return img
    else:
        raise TypeError('tiff file only support np.uint8, np.uint16, np.float, but got {}'.format(img.dtype))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # all the loader should be numpy ndarray [height, width, channels]
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img)


def image_loader(path):
    if os.path.splitext(path)[1].lower() in ['.tif', '.tiff']:
        return tifffile_loader(path)
    else:
        return pil_loader(path)


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


# def default_loader(path):
#     from torchvision import get_image_backend
#     if get_image_backend() == 'accimage':
#         return accimage_loader(path)
#     else:
#         return pil_loader(path)