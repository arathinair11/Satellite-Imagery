import os
os.system("sudo pip install rasterio")
os.system("sudo pip install pandas")
import rasterio
import pandas as pd
infile = r"/home/ubuntu/Capstone/Autoencoder/Cloud_free_Lagos_img.tif"
outfile = r'/home/ubuntu/Capstone/Autoencoder/Lagos/Tiff/Cloud_free_Lagos_img_{}.tif'
coordinates_path = r"/home/ubuntu/Capstone/Autoencoder/Lagos_Shapefile.csv"

N = 10
res = pd.read_csv(coordinates_path)
mapData = rasterio.open(infile)
with rasterio.open(infile) as dataset:

    # Loop through your list of coords
    for index, row in res.iterrows():

        # Get pixel coordinates from map coordinates
        py, px = dataset.index(row['long'], row['lat'])
        print('Pixel Y, X coords: {}, {}'.format(py, px))

        # Build an NxN window
        window = rasterio.windows.Window(px - N//2, py - N//2, N, N)
        print(window)

        # Read the data in the window
        # clip is a nbands * N * N numpy array
        clip = dataset.read(window=window)

        # You can then write out a new file
        meta = dataset.meta
        meta['width'], meta['height'] = N, N
        meta['transform'] = rasterio.windows.transform(window, dataset.transform)

        with rasterio.open(outfile.format(index), 'w', **meta) as dst:
            dst.write(clip)

