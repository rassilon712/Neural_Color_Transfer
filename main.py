import argparse
import os

import numpy as np
import torch
from torch import nn, optim
import torchvision.models
from torchvision import transforms, utils
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from utils import *


FEATURE_IDS = [1, 6, 11, 20, 29]
LEFT_SHIFT = (1, 2, 0)
RIGHT_SHIFT = (2, 0, 1)


def image_loader(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor


def get_feature(vgg19, img_tensor, feature_id):
    feature_tensor = vgg19.features[:feature_id](img_tensor)
    feature = feature_tensor.data.squeeze().cpu().numpy().transpose(LEFT_SHIFT)

    return feature


def normalize(feature):
    return feature / np.linalg.norm(feature, ord=2, axis=2, keepdims=True)


def main(config):
    device = torch.device(('cuda:' + str(config.gpu)) if config.cuda else 'cpu')

    imgS = image_loader(config.source).to(device)
    imgR = image_loader(config.reference).to(device)

    imgS_np = imgS.squeeze().numpy().transpose(LEFT_SHIFT)
    imgR_np = imgR.squeeze().numpy().transpose(LEFT_SHIFT)

    vgg19 = torchvision.models.vgg19(pretrained=True)
    vgg19.to(device)

    feat5S = get_feature(vgg19, imgS, FEATURE_IDS[4])
    feat5R = get_feature(vgg19, imgR, FEATURE_IDS[4])
    feat5S_norm = normalize(feat5S)
    feat5R_norm = normalize(feat5R)

    # FastGuidedFilter
    # labOrigS = torch.from_numpy(color.rgb2lab(np.array(origS)).transpose(RIGHT_SHIFT)).float()
    rgbOrigS = transforms.ToTensor()(origS)
    a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()
    b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Color Transfer between Images PyTorch")

    parser.add_argument('--source', type=str, default='./image/3_Source1', help="Source Image that has Content")
    parser.add_argument('--reference', type=str, default='./image/3_Reference', help="Reference Image to Get Style")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--processing_dir', type=str, default='./processImage')
    parser.add_argument('--cuda', dest='feature', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.set_defaults(cuda=False)
    # need more arguments?

    args = parser.parse_args()
    print(args)
    main(args)
