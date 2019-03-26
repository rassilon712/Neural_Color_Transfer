from PIL import Image
from skimage import color
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable

import numpy as np
import argparse

import utils


parser = argparse.ArgumentParser(description="Neural Color Transfer between Images PyTorch")
parser.add_argument('--source_image',type = str, default='image/3_Source1')
parser.add_arguement('--reference_image', type=str, default='image/3_Reference')
parser.add_argument("--cuda", dest='feature', action='store_true')
parser.set_defaults(cuda=False)



def main():
    args = parser.parse_args()
    #### should we put scale option?


