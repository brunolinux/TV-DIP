import numpy as np

def total_variation(img, reduction: str = "sum", step=1):
    pixel_dif1 = img[..., step:, :] - img[..., :-step, :]
    pixel_dif2 = img[..., :, step:] - img[..., :, :-step]

    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    reduce_axes = (-2, -1)
    if reduction == "mean":
        if img.is_floating_point():
            res1 = res1.to(img).mean(dim=reduce_axes)
            res2 = res2.to(img).mean(dim=reduce_axes)
        else:
            res1 = res1.float().mean(dim=reduce_axes)
            res2 = res2.float().mean(dim=reduce_axes)
    elif reduction == "sum":
        res1 = res1.sum(dim=reduce_axes)
        res2 = res2.sum(dim=reduce_axes)

    return res1 + res2