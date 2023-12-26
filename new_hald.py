import math
import numpy as np
import imageio.v2 as imageio


def generate_hald(size=33):
    res = math.ceil(max(math.sqrt(size ** 3) / size, 256. / size)) * size
    img = np.zeros((res, res, 3), dtype='float64')
    counter = 0
    for b in range(size):
        for g in range(size):
            for r in range(size):
                img[counter // res, counter % res] = np.array((r, g, b))
                counter += 1
    img = ((img * (1 / (size - 1.))) * (2 ** 16 - 1)).astype(dtype='uint16')
    imageio.imsave("hald.tif", img)


generate_hald(size=65)