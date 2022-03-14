## Mapping Deprived Areas In Low & Middle Income Countries Using Satellite Images

This repository consist of Python code for extracting cloud-free Sentinel-2 images for Lagos, Accra and Nairobi from Google Earth Engine and a CNN Autoencoder for image classification.

### Steps for executing the cloud-free Sentinel-2 code:
Create a [Google Earth Engine](https://earthengine.google.com) account before running the code.
##### Library Installations:
* pip install earthengine-api
* pip install folium
##### Parameters chosen to fine-tune the image:
* START_DATE 
* END_DATE 
* CLOUD_FILTER 
* CLD_PRB_THRESH 
* CLD_PRJ_DIST 
* BUFFER

### For CNN Autoencoder code, connect to AWS server and execute the .py file. 
* CNN Autoencoder.py - Trained using satellite images extracted from the training dataset coordinates(coordinates.csv)
* CNN_Autoencoder_Unlabeled.py - Trained using satellite images extracted randomly using the shapefile coordinates(Lagos_Shapefile.csv)

##### References:
* [Sentinel-2 Cloud Masking with s2cloudless](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless)
* [Exporting Image](https://colab.research.google.com/github/csaybar/EEwPython/blob/dev/10_Export.ipynb#scrollTo=M9EbU74_ESvY)
