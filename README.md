# TV-DIP: Total variation with deep image prior 
![](img/struct.png)

The core idea behind this paper



# Result 
![](img/video.png)

# Install
```shell
git clone https://github.com/brunolinux/TV-DIP.git
virtualenv env_tv_dip --python==3.8
source <path to env_tv_dip>/bin/activate
pip install -r requirements.txt
```

**note:** pytorch is install under the instructions from the official website: https://pytorch.org/get-started/locally/

# Dataset
you have to use the script `create_noisy_data.py` to create noisy image

Three types of noise pattern is supported:

- "lpn": low-frequency pattern noise
- "hpn": high-frequency pattern noise
- "combined": "lpn" + "hpn"

the noise strength and other parameters is defined in this file



# Train

jupyter is required here

- `1_fpn_denoise.ipynb` is used to test on high-frequency pattern noise
- `2_lpn_denoise.ipynb` is used to test on low-frequency pattern noise
- `3_spn_denoise.ipynb` is used to test on mixed noise
- `4_video.ipynb` is used to test on video frames.






# Citation
Kang Liu, Honglei Chen, Wenzhong Bao, Jianlu Wang,
Thermal imaging spatial noise removal via deep image prior and step-variable total variation regularization,
Infrared Physics & Technology,
https://doi.org/10.1016/j.infrared.2023.104888.
(https://www.sciencedirect.com/science/article/pii/S1350449523003468)
Abstract: The imaging quality of an uncooled long wave infrared camera suffers severely from time-drifted spatial noises if not regulated by a thermoelectric cooler (TEC). These noises can be categorized within the frequency domain as either fixed pattern noise (FPN) or low-frequency noise (LPN). The time-drifted FPN is primarily governed by the temperature fluctuations of the focal plane array (FPA), whereas the LPN predominantly originates from non-uniform background radiations. Most existing methods have been proposed to address the FPN or LPN independently rather than from a unified perspective. In this paper, a self-supervised convolution neural network (CNN) is proposed to remove these two types of noise simultaneously. Because of the combination of the deep image prior (DIP) architecture and the total variation (TV) regularizer, this method is named TV-DIP. With the assumption that the spatial noises are constant in a short period of time, the spatial noise pattern is extracted directly from multiple continuous raw frames without ground-truth noise-free images. In this way, the noise pattern can be updated periodically during running time, without requiring the use of a mechanical shutter. Besides, the original TV regularizer is extended with a variable step, which shows a significant enhancement on the LPN denoising task. Furthermore, both simulation and experimental results have demonstrated that this approach achieves better performance in terms of the removal of spatial noise, the preservation of detail, and the suppression of artifacts. The source code and dataset are available on the website: https://github.com/brunolinux/TV-DIP
Keywords: Infrared imaging; Spatial noise removal; Deep image prior; Total variation regularizer
