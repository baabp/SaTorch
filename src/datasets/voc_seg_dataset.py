"""
    Pascal VOC2012 semantic segmentation dataset.
"""

import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.datasets.seg_dataset import SegDataset
from src.datasets.dataset_metainfo import DatasetMetaInfo

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

class VOCSegDataset(SegDataset):
    """
    Pascal VOC2012 semantic segmentation dataset.

    Parameters
    ----------
    root : str
        Path to VOCdevkit folder.
    mode : str, default 'train'
        'train', 'val', 'test', or 'demo'.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 root='data',
                 mode="train",
                 transform=None,
                 **kwargs):
        super(VOCSegDataset, self).__init__(
            root=root,
            mode=mode,
            transform=transform,
            **kwargs)

        base_dir_path = os.path.join(root, "VOC2012")
        image_dir_path = os.path.join(base_dir_path, "JPEGImages")
        mask_dir_path = os.path.join(base_dir_path, "SegmentationClass")

        splits_dir_path = os.path.join(base_dir_path, "ImageSets", "Segmentation")
        if mode == "train":
            split_file_path = os.path.join(splits_dir_path, "train.txt")
        elif mode in ("val", "test", "demo"):
            split_file_path = os.path.join(splits_dir_path, "val.txt")
        else:
            raise RuntimeError("Unknown dataset splitting mode")

        self.images = []
        self.masks = []
        with open(os.path.join(split_file_path), "r") as lines:
            for line in lines:
                image_file_path = os.path.join(image_dir_path, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(image_file_path)
                self.images.append(image_file_path)
                mask_file_path = os.path.join(mask_dir_path, line.rstrip('\n') + ".png")
                assert os.path.isfile(mask_file_path)
                self.masks.append(mask_file_path)

        assert (len(self.images) == len(self.masks))

    # def __getitem__(self, index):
    #     img = image_loader(self.images[index])
    #     target = image_loader(self.masks[index])
    #
    #     if self.mode == "train":
    #         img, target = self._sync_transform(img, target)
    #     elif self.mode == "val":
    #         img, target = self._val_sync_transform(img, target)
    #     else:
    #         assert self.mode == "test"
    #         img, target = self._img_transform(img), self._mask_transform(target)
    #
    #     if self.transform is not None:
    #         img, target = self.transform(img, target)
    #     return img, target
    #
    # def __len__(self):
    #     return len(self.images)


    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.mode == "demo":
            image = self._img_transform(image)
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])

        if self.mode == "train":
            image, mask = self._sync_transform(image, mask)
        elif self.mode == "val":
            image, mask = self._val_sync_transform(image, mask)
        else:
            assert self.mode == "test"
            image, mask = self._img_transform(image), self._mask_transform(mask)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    classes = 21
    vague_idx = 255
    use_vague = True
    background_idx = 0
    ignore_bg = True

    @staticmethod
    def _mask_transform(mask):
        np_mask = np.array(mask).astype(np.int32)
        # np_mask[np_mask == 255] = VOCSegDataset.vague_idx
        return np_mask

    def __len__(self):
        return len(self.images)


def voc_test_transform(ds_metainfo,
                       mean_rgb=(0.485, 0.456, 0.406),
                       std_rgb=(0.229, 0.224, 0.225)):
    assert (ds_metainfo is not None)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])


class VOCMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(VOCMetaInfo, self).__init__()
        self.label = "VOC"
        self.short_label = "voc"
        self.root_dir_name = "voc"
        self.dataset_class = VOCSegDataset
        self.num_training_samples = None
        self.in_channels = 3
        self.num_classes = VOCSegDataset.classes
        self.input_image_size = (480, 480)
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.test_metric_extra_kwargs = [{}, {}]
        self.test_metric_capts = ["Val.PixAcc", "Val.IoU"]
        self.test_metric_names = ["PixelAccuracyMetric", "MeanIoUMetric"]
        self.test_metric_extra_kwargs = [
            {"vague_idx": VOCSegDataset.vague_idx,
             "use_vague": VOCSegDataset.use_vague,
             "macro_average": False},
            {"num_classes": VOCSegDataset.classes,
             "vague_idx": VOCSegDataset.vague_idx,
             "use_vague": VOCSegDataset.use_vague,
             "bg_idx": VOCSegDataset.background_idx,
             "ignore_bg": VOCSegDataset.ignore_bg,
             "macro_average": False}]
        self.saver_acc_ind = 1
        self.train_transform = None
        self.val_transform = voc_test_transform
        self.test_transform = voc_test_transform
        self.ml_type = "imgseg"
        self.allow_hybridize = False
        self.net_extra_kwargs = {"aux": False, "fixed_size": False}
        self.load_ignore_extra = True
        self.image_base_size = 520
        self.image_crop_size = 480

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(VOCMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--image-base-size",
            type=int,
            default=520,
            help="base image size")
        parser.add_argument(
            "--image-crop-size",
            type=int,
            default=480,
            help="crop image size")

    def update(self,
               args):
        super(VOCMetaInfo, self).update(args)
        self.image_base_size = args.image_base_size
        self.image_crop_size = args.image_crop_size


def main():
    dataset = VOCSegDataset(root='data',
                            mode="train",
                            transform=None)
    print('flag')

if __name__ == "__main__":
    main()
