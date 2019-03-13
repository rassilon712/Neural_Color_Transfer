# Neural_Color_Transfer_PyTorch_Implementation
We are trying to implement neural color transfer by Mingming He et al in PyTorch.

# Process
Implemented on Single Reference Neural Color Transfer
And changed WLS filter to Deep Guided Filter.

# Results
Input Imag
![Input](https://user-images.githubusercontent.com/15609545/54282220-e0ea0480-45de-11e9-810f-8d8467c0b60e.jpg)

Style image
![Style](https://user-images.githubusercontent.com/15609545/54282236-e8111280-45de-11e9-862c-a8c90724de85.jpg)

And the result from those images are here. The images are from layer 5 to 1 below.

L=5 
![img5S](https://user-images.githubusercontent.com/15609545/54282066-8355b800-45de-11e9-9765-4071bf432e19.png)

L=4
![img4S](https://user-images.githubusercontent.com/15609545/54282069-851f7b80-45de-11e9-8bf6-7dd66ab3c9ac.png)

L=3
![img3S](https://user-images.githubusercontent.com/15609545/54282074-8781d580-45de-11e9-8c9b-c1f701736da5.png)

L=2
![img2S](https://user-images.githubusercontent.com/15609545/54282080-894b9900-45de-11e9-9192-af2c2ed814d6.png)

L=1
![img1S](https://user-images.githubusercontent.com/15609545/54282082-8badf300-45de-11e9-8b8b-9deeab193846.png)

# TODOS
 - Need to modularize our notebook implementation.
 - Our guidance and result image is different from original paper so need to fix some codes.
 - The performance speed is way too slow than original paper.
 - If the single reference is working well, need to work on multi-reference.
 - Refine MarkDown
