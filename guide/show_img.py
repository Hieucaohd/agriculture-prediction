# ipython --pylab

import spectral.io.envi as envi
img = envi.open("./data/spectral_image/hyper_20220913_3cm.hdr", "./data/spectral_image/hyper_20220913_3cm.img")
from spectral import open_image, imshow
view = imshow(img)
