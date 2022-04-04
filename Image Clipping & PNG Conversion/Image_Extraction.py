import os.path
from itertools import product
import rasterio as rio
from rasterio import windows
import matplotlib.image as mpimg
import os
import numpy as np
import skimage
from PIL import Image


infile = '/home/ubuntu/Capstone/Autoencoder/Images/Cloud_free_Nairobi_img.tif'
out_path = '/home/ubuntu/Capstone/Autoencoder/Nairobi/TIF'
output_filename = 'nairobi_{}-{}.tif'

# extract 10 x 10px  images from tif
def get_tiles(ds, width=10, height=10):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


with rio.open(os.path.join(infile)) as inds:
    tile_width, tile_height = 10, 10

    meta = inds.meta.copy()

    for window, transform in get_tiles(inds):
        print(window)
        meta['transform'] = transform
        meta['width'], meta['height'] = window.width, window.height
        outpath = os.path.join(out_path,output_filename.format(int(window.col_off), int(window.row_off)))
        with rio.open(outpath, 'w', **meta) as outds:
            outds.write(inds.read(window=window))

# remove images less than 10 x 10 px
img_dir = "/home/ubuntu/Capstone/Autoencoder/Nairobi/PNG/Train"
for filename in os.listdir(img_dir):
    filepath = os.path.join(img_dir, filename)
    with Image.open(filepath) as im:
        x, y = im.size
    totalsize = x*y
    if totalsize < 100:
        os.remove(filepath)

# delete black/blank images i.e. less than 0.25
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        img = mpimg.imread(filepath)
        img = skimage.img_as_float(img)
        print(np.mean(img))
        # if img is not None:
        #     images.append(img)
        if(np.mean(img) <= 0.25):
            os.remove(filepath)

load_images('/home/ubuntu/Capstone/Autoencoder/Nairobi/PNG/Train')

