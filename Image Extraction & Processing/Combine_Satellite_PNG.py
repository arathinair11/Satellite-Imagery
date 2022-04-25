# importing required packages
from pathlib import Path
import shutil
import os

# Combine all three cities PNG images into one folder

parent_dir = '/home/ubuntu/Autoencoder/Final'
directory = 'Satellite_Images/Train/'

folder_path = os.path.join(parent_dir, directory)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# defining source and destination paths
name = 'Accra'
src = f'/home/ubuntu/Autoencoder/{name}/PNG/Train'
trg = folder_path

for src_file in Path(src).glob('*.*'):
    shutil.copy(src_file, trg)

print("Transfer Complete")

