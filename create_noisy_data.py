import numpy as np
from PIL import Image 
import os 
import cv2
from matplotlib import pyplot as plt

base_dir = "./data/human"
in_dir = "images_clean"
out_dir = "images_combined"
noise_file = os.path.join(base_dir, "..", "lf_noise_1.jpg")
noise_out_np_file = os.path.join(base_dir, "noise_combined.npz")
noise_out_img_file = os.path.join(base_dir, "noise_combined.png")
shape = (480, 640)

def create_low_frequency_noise(file_path, shape, max_strength = 0.5):
    noise = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    print(noise.shape)

    noise = cv2.resize(noise, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    noise = np.array(noise, dtype=np.float32)
    noise = noise / 256 * max_strength
    return noise

def create_column_strip(random_generator: np.random.Generator, shape, sigma_white, sigma_strip):
    white_noise = random_generator.random(shape) * sigma_white
    strip_column = random_generator.random((1, shape[1])) * sigma_strip
    strip = np.repeat(strip_column, shape[0], axis=0)
    noise = strip + white_noise
    return noise



def create_image_noise(noise_type:str):
    rng = np.random.default_rng(1998) # 2022
    sigma_white = 0.1
    sigma_strip = 0.15
    lf_max_strength = 0.4
    image_strength = 0.7

    if noise_type == "lpn":
        noise = create_low_frequency_noise(noise_file, shape, lf_max_strength)
    elif noise_type == "fpn":
        noise = create_column_strip(rng, shape, sigma_white, sigma_strip)
    elif noise_type == "combined":
        noise_lf = create_low_frequency_noise(noise_file, shape, lf_max_strength)
        noise_fpn = create_column_strip(rng, shape, sigma_white, sigma_strip)
        noise = noise_lf + noise_fpn

        # noise_img = np.array(noise_lf /np.max(noise_lf) * 255, dtype=np.uint8)
        # cv2.imwrite(os.path.join(base_dir, "noise_lf_out.png"), noise_img)
        # noise_img = np.array(noise_fpn/np.max(noise_fpn) * 255, dtype=np.uint8)
        # cv2.imwrite(os.path.join(base_dir, "noise_fpn_out.png"), noise_img)

    else:
        print("noise type {} is not supported".format(noise_type))
        return

    noise = np.expand_dims(noise, 2)

    files = os.listdir(os.path.join(base_dir, in_dir))
    
    for file in files:
        path = os.path.join(base_dir, in_dir, file)
        image = np.array(Image.open(path), dtype=np.float32)
        image = image / 255 * image_strength
        image = image + noise
        count = np.sum(image > 1)
        image = image / np.max(image)
        image = image * 255
        image = np.asarray(image, dtype=np.uint8)
        image = Image.fromarray(image)
        image.save(os.path.join(base_dir, out_dir, file))
        print("{} overflow count: {}!".format(path, count))


    # np.save(noise_out_np_file, noise)
    # noise_img = np.array(noise * 255, dtype=np.uint8)
    # cv2.imwrite(noise_out_img_file, noise_img)

    plt.imshow(noise[..., 0])
    plt.show()



def create_random_noise():
    rng = np.random.default_rng(2022)
    white_noise = rng.random(shape)
    noise_img = np.array(white_noise /np.max(white_noise) * 255, dtype=np.uint8)
    cv2.imwrite(os.path.join(base_dir, "noise_random_out.png"), noise_img)


if __name__ == "__main__":
    create_image_noise("combined") # "fpn"/ "combined" / "lpn"
    # create_random_noise()