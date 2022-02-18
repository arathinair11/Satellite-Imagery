# Cloud-Free Satellite Image Preprocessing

This repository consist of Python code for extracting cloud-free Sentinel-2 images for Lagos, Accra and Nairobi from Google Earth Engine.

### Running the Code
Create a [Google Earth Engine](https://earthengine.google.com) account before running the code.

#### Library Installations:
* pip install earthengine-api

* pip install folium

#### Parameters chosen to fine-tune the image:

* START_DATE 
* END_DATE 
* CLOUD_FILTER 
* CLD_PRB_THRESH 
* CLD_PRJ_DIST 
* BUFFER

#### References:

* [Sentinel-2 Cloud Masking with s2cloudless](https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless)

* [Exporting Image](https://colab.research.google.com/github/csaybar/EEwPython/blob/dev/10_Export.ipynb#scrollTo=M9EbU74_ESvY)
