import argparse
import os

import numpy as np
import torch
import torchvision.models
from torchvision import transforms, utils
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans

from models import LocalColorTransfer
from utils import *


FEATURE_IDS = [1, 6, 11, 20, 29]
LEFT_SHIFT = (1, 2, 0)
RIGHT_SHIFT = (2, 0, 1)


def image_loader(img_path):
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor


def resize_img(img, size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)

    return img_tensor


def get_feature(vgg19, img_tensor, feature_id):
    feature_tensor = vgg19.features[:feature_id](img_tensor)
    feature = feature_tensor.data.squeeze().cpu().numpy().transpose(LEFT_SHIFT)

    return feature


def normalize(feature):
    return feature / np.linalg.norm(feature, ord=2, axis=2, keepdims=True)


def main(config):
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.processing_dir):
        os.makedirs(config.processing_dir)

    device = torch.device(('cuda:' + str(config.gpu)) if config.cuda else 'cpu')

    vgg19 = torchvision.models.vgg19(pretrained=True)
    vgg19.to(device)

    origS = Image.open(config.source).convert("RGB")
    origR = Image.open(config.reference).convert("RGB")

    imgS = image_loader(config.source).to(device)
    imgR = image_loader(config.reference).to(device)

    imgS_np = imgS.squeeze().numpy().transpose(LEFT_SHIFT)
    imgR_np = imgR.squeeze().numpy().transpose(LEFT_SHIFT)

    feat5S = get_feature(vgg19, imgS, FEATURE_IDS[4])
    feat5R = get_feature(vgg19, imgR, FEATURE_IDS[4])
    feat5S_norm = normalize(feat5S)
    feat5R_norm = normalize(feat5R)

    map5SR = PatchMatch(feat5S_norm, feat5R_norm)  # S -> R
    map5RS = PatchMatch(feat5R_norm, feat5S_norm)  # R -> S
    map5SR.solve()
    print()
    map5RS.solve()

    imgS_resized = resize_img(origS, feat5S.shape[:2])
    imgR_resized = resize_img(origR, feat5R.shape[:2])

    imgG = bds_vote(imgR_resized, map5SR.nnf, map5RS.nnf)
    feat5G = bds_vote(feat5R.transpose(RIGHT_SHIFT), map5SR.nnf, map5RS.nnf).transpose(LEFT_SHIFT)
    feat5G_norm = normalize(feat5G)

    # Bookmark
    kmeans = KMeans(n_clusters=5, n_jobs=1).fit(feat5S.reshape(-1, feat5S.shape[2]))
    kmeans_labels = kmeans.labels_.reshape(feat5S.shape[:2])

    labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
    labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

    lct = LocalColorTransfer(imgS_resized.numpy().transpose(LEFT_SHIFT), imgG.transpose(LEFT_SHIFT),
                             feat5S_norm, feat5G_norm, kmeans_labels, device, kmeans_ratio=1)
    save = torch.from_numpy(imgG).float()
    utils.save_image(save, config.result_dir + 'img5G.png')

    # FastGuidedFilter
    # labOrigS = torch.from_numpy(color.rgb2lab(np.array(origS)).transpose(RIGHT_SHIFT)).float()
    rgbOrigS = transforms.ToTensor()(origS)
    a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()
    b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()

    img5S = a_upsampled * rgbOrigS + b_upsampled
    img5S = img5S.data.numpy().transpose(LEFT_SHIFT).astype(np.float64)
    # img5S = color.lab2rgb(img5S.data.numpy().transpose(LEFT_SHIFT).astype(np.float64))
    # imshow(img5S)

    img5S = torch.from_numpy(img5S.transpose(RIGHT_SHIFT)).float()
    utils.save_image(img5S, config.result_dir + 'img5S.png')
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img5S)
    img5S = img5S.unsqueeze(0)

    # ------------------------------------------------------------------------------------------------------------------

    feat4S = get_feature(vgg19, img5S, FEATURE_IDS[3])
    feat4R = get_feature(vgg19, imgR, FEATURE_IDS[3])
    feat4S_norm = normalize(feat4S)
    feat4R_norm = normalize(feat4R)

    map4SR = PatchMatch(feat4S_norm, feat4R_norm) #S -> R
    map4RS = PatchMatch(feat4R_norm, feat4S_norm) #R -> S
    map4SR.solve()
    print()
    map4RS.solve()

    imgS_resized = resize_img(origS, feat4S.shape[:2])
    imgR_resized = resize_img(origR, feat4R.shape[:2])

    imgG = bds_vote(imgR_resized, map4SR.nnf, map4RS.nnf)
    feat4G = bds_vote(feat4R.transpose(RIGHT_SHIFT), map4SR.nnf, map4RS.nnf).transpose(LEFT_SHIFT)
    feat4G_norm = normalize(feat4G)

    labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
    labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

    lct = LocalColorTransfer(imgS_resized.numpy().transpose(LEFT_SHIFT), imgG.transpose(LEFT_SHIFT),
                             feat4S_norm, feat4G_norm, kmeans_labels, device, kmeans_ratio=2)
    save = torch.from_numpy(imgG).float()
    utils.save_image(save, config.result_dir + 'img4G.png')
    lct.train()

    a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()
    b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()

    img4S = a_upsampled * rgbOrigS + b_upsampled
    img4S = img4S.data.numpy().transpose(LEFT_SHIFT)
    # imshow(img4S)

    img4S = torch.from_numpy(img4S.transpose(RIGHT_SHIFT)).float()
    utils.save_image(img4S, config.result_dir + 'img4S.png')
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img4S)
    img4S = img4S.unsqueeze(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Color Transfer between Images PyTorch")

    parser.add_argument('--source', type=str, default='./image/3_Source1', help="Source Image that has Content")
    parser.add_argument('--reference', type=str, default='./image/3_Reference', help="Reference Image to Get Style")
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--processing_dir', type=str, default='./processImage')
    parser.add_argument('--cuda', dest='feature', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.set_defaults(cuda=False)
    # need more arguments?

    args = parser.parse_args()
    print(args)
    main(args)
