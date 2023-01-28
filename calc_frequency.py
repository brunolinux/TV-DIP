import numpy as np
import os 
import cv2
from matplotlib import pyplot as plt


base_dir = "./data/power_supply"
img_name = "0025.png"
img_clean_file = os.path.join(base_dir, "images_clean", img_name) 
img_fpn_file = os.path.join(base_dir, "images_fpn", img_name) 
img_lpn_file = os.path.join(base_dir, "images_lpn_2", img_name) 
img_spn_file = os.path.join(base_dir, "images_combined", img_name) 

def load_image(file: str):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.float32) / 255
    return img

def calc_freq_histogram(img, range, bins=1000):
    f = np.fft.fft2(img)
    mag = np.log(np.abs(f))
    mag_hist, _ = np.histogram(mag.flatten(), bins=bins, density=True, range=range)
    return mag_hist

if __name__ == "__main__":
    img_clean = load_image(img_clean_file)
    img_fpn = load_image(img_fpn_file)
    img_lpn = load_image(img_lpn_file)
    img_spn = load_image(img_spn_file)

    bins = 2000
    range = (-6, 6)
    x = np.linspace(start=range[0], stop=range[1], num=bins)

    freq_clean = calc_freq_histogram(img_clean, range, bins)
    freq_fpn = calc_freq_histogram(img_fpn, range, bins)
    freq_lpn = calc_freq_histogram(img_lpn, range, bins)
    freq_spn = calc_freq_histogram(img_spn, range, bins)

    if True:

        fig = plt.figure(num=None, figsize=(4, 3), dpi=300)
        plt.plot(x, freq_clean, color="#8ECFC9", label="clean (a)")
        plt.plot(x, freq_fpn, color="#FFBE7A", label="FPN (b)")
        plt.plot(x, freq_lpn, color="#FA7F6F", label="LPN (c)")
        plt.plot(x, freq_spn, color="#82B0D2", label="FPN + LPN (d)")
        plt.legend(loc='upper left', prop={'size': 8})
        xticks = np.arange(range[0], range[1]+1, step=2)
        plt.xticks(xticks, fontsize=12)
        plt.yticks(np.linspace(0, 0.7, num=8), fontsize=12)
        plt.xlabel(u'frequency (log space)', fontsize=12)
        plt.ylabel(u'probility density', fontsize=12)
        plt.savefig('./figure_freq.png', bbox_inches='tight')
        plt.show()

