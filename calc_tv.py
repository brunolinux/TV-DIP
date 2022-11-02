import numpy as np
import os 
import cv2
from matplotlib import pyplot as plt


base_dir = "./data/power_supply"
img_name = "0025.png"
img_fpn_file = os.path.join(base_dir, "images_fpn", img_name) 
img_lpn_file = os.path.join(base_dir, "images_lpn_2", img_name) 
img_spn_file = os.path.join(base_dir, "images_combined", img_name) 

def load_image(file: str):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=np.float32) / 255
    return img

def total_variation_numpy(img, reduction: str = "sum", step=1):
    pixel_dif1 = img[step:, :] - img[:-step, :]
    pixel_dif2 = img[:, step:] - img[:, :-step]

    res1 = np.abs(pixel_dif1)
    res2 = np.abs(pixel_dif2)

    reduce_axes = (-2, -1)
    if reduction == "mean":
        res1 = np.mean(res1, axis=reduce_axes)
        res2 = np.mean(res2, axis=reduce_axes)
    elif reduction == "sum":
        res1 = np.sum(res1, axis=reduce_axes)
        res2 = np.sum(res2, axis=reduce_axes)

    return res1 + res2


def calc_tv_range(img, min_v, max_v):
    val_list = []
    for step in range(min_v, max_v):
        img_blur = cv2.blur(img, (step+2,step+2))
        val = total_variation_numpy(img_blur, "sum", step)
        val_list.append(val)

    ratio_list = []
    for val in val_list:
        ratio = val / val_list[0]
        ratio_list.append(ratio)

    return val_list, ratio_list


if __name__ == "__main__":
    img_fpn = load_image(img_fpn_file)
    img_lpn = load_image(img_lpn_file)
    img_spn = load_image(img_spn_file)

    fpn_tv_list, r_fpn_tv_list = calc_tv_range(img_fpn, 1, 80)
    lpn_tv_list, r_lpn_tv_list = calc_tv_range(img_lpn, 1, 80)
    spn_tv_list, r_spn_tv_list = calc_tv_range(img_spn, 1, 80)


    if True:
        step = 5
        s_fpn_tv_list = r_fpn_tv_list[::step]
        s_lpn_tv_list = r_lpn_tv_list[::step]
        s_spn_tv_list = r_spn_tv_list[::step]
        s_x = np.arange(1, 80, step)
        fig = plt.figure(num=None, figsize=(4, 3), dpi=300)
        plt.plot(s_x, s_fpn_tv_list, color="#8ECFC9", marker='o', label="fpn")
        plt.plot(s_x, s_lpn_tv_list, color="#FFBE7A", marker='s', label="lpn")
        plt.plot(s_x, s_spn_tv_list, color="#FA7F6F", marker='+', label="fpn + lpn")
        plt.legend()
        plt.xticks(np.arange(81, step=10), fontsize=12)
        plt.yticks(np.arange(19, step=3), fontsize=12)
        plt.xlabel(u'step', fontsize=12)
        plt.ylabel(u'TV ratio', fontsize=12)
        plt.savefig('./figure.png', bbox_inches='tight')
        plt.show()


    # fig1, axes = plt.subplots(1, 3, figsize=(18,6))

    # axes[0].imshow(img_fpn)
    # axes[1].imshow(img_lpn)
    # axes[2].imshow(img_spn)

    # plt.show()