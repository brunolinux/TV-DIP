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
