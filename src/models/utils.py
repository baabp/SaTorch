
"""
    Main routines shared between training and evaluation scripts.
"""

import logging
import os
import numpy as np
import torch.utils.data
from .pytorchcv.model_provider import get_model
# from .metrics.metric import EvalMetric, CompositeEvalMetric
# from .metrics.cls_metrics import Top1Error, TopKError
# from .metrics.seg_metrics import PixelAccuracyMetric, MeanIoUMetric
# from .metrics.det_metrics import CocoDetMApMetric
# from .metrics.hpe_metrics import CocoHpeOksApMetric

from src.torchsat.torchsat.models.classification.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
from src.torchsat.torchsat.models.classification.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from src.torchsat.torchsat.models.classification.densenet import densenet121, densenet169, densenet201
from src.torchsat.torchsat.models.classification.inception import inception_v3
from src.torchsat.torchsat.models.classification.mobilenet import mobilenet_v2
from src.torchsat.torchsat.models.classification.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from src.torchsat.torchsat.models.classification.resnest import resnest50, resnest101, resnest200, resnest269
from src.torchsat.torchsat.models.segmentation.unet import unet34, unet101, unet152


models = {
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
    'vgg19': vgg19,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'mobilenet_v2': mobilenet_v2,
    'inception_v3': inception_v3,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,
    'efficientnet_b2': efficientnet_b2,
    'efficientnet_b3': efficientnet_b3,
    'efficientnet_b4': efficientnet_b4,
    'efficientnet_b5': efficientnet_b5,
    'efficientnet_b6': efficientnet_b6,
    'efficientnet_b7': efficientnet_b7,
    'resnest50': resnest50,
    'resnest101': resnest101,
    'resnest200': resnest200,
    'resnest269': resnest269,

    'unet34': unet34,
    'unet101': unet101,
    'unet152': unet152,
}

def get_model_torchsat(name: str, num_classes: int, **kwargs):
    print(kwargs)
    if name.lower() not in models:
        raise ValueError("no model named {}, should be one of {}".format(name, ' '.join(models)))

    return models.get(name.lower())(num_classes, **kwargs)


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_cuda,
                  use_data_parallel=True,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  num_classes=None,
                  in_channels=None,
                  remap_to_cpu=False,
                  remove_module=False):
    """
    Create and initialize model by name.
    Parameters
    ----------
    model_name : str
        Model name.
    use_pretrained : bool
        Whether to use pretrained weights.
    pretrained_model_file_path : str
        Path to file with pretrained weights.
    use_cuda : bool
        Whether to use CUDA.
    use_data_parallel : bool, default True
        Whether to use parallelization.
    net_extra_kwargs : dict, default None
        Extra parameters for model.
    load_ignore_extra : bool, default False
        Whether to ignore extra layers in pretrained model.
    num_classes : int, default None
        Number of classes.
    in_channels : int, default None
        Number of input channels.
    remap_to_cpu : bool, default False
        Whether to remape model to CPU during loading.
    remove_module : bool, default False
        Whether to remove module from loaded model.
    Returns
    -------
    Module
        Model.
    """
    kwargs = {"pretrained": use_pretrained}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        checkpoint = torch.load(
            pretrained_model_file_path,
            map_location=(None if use_cuda and not remap_to_cpu else "cpu"))
        if (type(checkpoint) == dict) and ("state_dict" in checkpoint):
            checkpoint = checkpoint["state_dict"]

        if load_ignore_extra:
            pretrained_state = checkpoint
            model_dict = net.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
            net.load_state_dict(pretrained_state)
        else:
            if remove_module:
                net_tmp = torch.nn.DataParallel(net)
                net_tmp.load_state_dict(checkpoint)
                net.load_state_dict(net_tmp.module.cpu().state_dict())
            else:
                net.load_state_dict(checkpoint)

    if use_data_parallel and use_cuda:
        net = torch.nn.DataParallel(net)

    if use_cuda:
        net = net.cuda()

    return net