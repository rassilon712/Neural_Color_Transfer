# Neural_Color_Transfer_PyTorch_Implementation
We are trying to implement [Neural Color Transfer between Images][1] by Mingming He et al in PyTorch.
They revised their paper to version 2 which is [Progressive Color Transfer with Dense Semantic Correspondences][2].

## Paper
### Progressive Color Transfer with Dense Semantic Correspondences
#### Abstract
We propose a new algorithm for color transfer between images that have perceptually similar semantic structures. We aim to achieve a more accurate color transfer that leverages semantically-meaningful dense correspondence between images. To accomplish this, our algorithm uses neural representations for matching. Additionally, the color transfer should be spatially variant and globally coherent. Therefore, our algorithm optimizes a local linear model for color transfer satisfying both local and global constraints. Our proposed approach jointly optimizes matching and color transfer, adopting a coarse-to-fine strategy. The proposed method can be successfully extended from one-to-one to one-to-many color transfer. The latter further addresses the problem of mismatching elements of the input image. We validate our proposed method by testing it on a large variety of image content.

[1]:https://arxiv.org/abs/1710.00756v1
[2]:https://arxiv.org/abs/1710.00756v2

## Process
Implemented on Single Reference Neural Color Transfer
And changed WLS-based filter to [Deep Guided Filter][3].


### Pipeline
![K-006](https://user-images.githubusercontent.com/15609545/54339117-37a31d00-4677-11e9-9bb3-a6fa0ee48fd1.jpg)




[3]:https://github.com/wuhuikai/DeepGuidedFilter

## Results
### Input Image
![Input](https://user-images.githubusercontent.com/15609545/54282220-e0ea0480-45de-11e9-810f-8d8467c0b60e.jpg)

### Style image
![Style](https://user-images.githubusercontent.com/15609545/54282236-e8111280-45de-11e9-862c-a8c90724de85.jpg)

And the result from those images are here. The images are from layer 5 to 1 below.

### L=5 
![img5S](https://user-images.githubusercontent.com/15609545/54282066-8355b800-45de-11e9-9765-4071bf432e19.png)

### L=4
![img4S](https://user-images.githubusercontent.com/15609545/54282069-851f7b80-45de-11e9-8bf6-7dd66ab3c9ac.png)

### L=3
![img3S](https://user-images.githubusercontent.com/15609545/54282074-8781d580-45de-11e9-8c9b-c1f701736da5.png)

### L=2
![img2S](https://user-images.githubusercontent.com/15609545/54282080-894b9900-45de-11e9-9192-af2c2ed814d6.png)

### L=1
![img1S](https://user-images.githubusercontent.com/15609545/54282082-8badf300-45de-11e9-8b8b-9deeab193846.png)

## TODO
 - Need to modularize our notebook implementation.
 - Our guidance and result image is different from original paper so need to fix some codes.
 - The performance speed is way too slow than original paper.
 - If the single reference is working well, need to work on multi-reference.
 - Refine MarkDown
