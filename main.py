from PIL import Image
from skimage import color
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable
from torch import nn, optim
from torchvision import models, transforms, utils
from guided_filter_pytorch.guided_filter import FastGuidedFilter
import numpy as np
import argparse
import os

import utils


parser = argparse.ArgumentParser(description="Neural Color Transfer between Images PyTorch")
parser.add_argument('--source_image',type = str, default='image/3_Source1', help= "Source Image that has Content")
parser.add_arguement('--reference_image', type=str, default='image/3_Reference', help= "Reference Image to Get Style")
parser.add_arguement('--results_path',type=str, default='/results')
parser.add_arguement('--processing_path', type=str, default='/processimage')
parser.add_arguement('--gpu', type=int, default=0)
parser.add_argument("--cuda", dest='feature', action='store_true')
parser.set_defaults(cuda=False)

## need more arguements?


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))

    # FastGuidedFilter
    # labOrigS = torch.from_numpy(color.rgb2lab(np.array(origS)).transpose(RIGHT_SHIFT)).float()
    rgbOrigS = transforms.ToTensor()(origS)
    a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()
    b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                                 rgbOrigS.unsqueeze(0)).squeeze()

