This module contains the code required to do OCR of product NFTs

### Details

This uses tesseract to extract nutrition information out of images of NFTs. There are a number of preprocessing steps that are worth describing:

1. First, there is the cropper, that will crop the image out of a larger image of the whole package. This is located in NFT_PreProcessing.NFTBoxDetection
2. There is the image unwrapping code (ImageUnwrapping) which is used to unwrap the curvature of images on curved surfaces like soup cans. It does check to see if any modification is required, so it won't make things worse, but it will slow things down a fair bit, so don't use it if it's not necessary.
3. Finally, there is a general purpose preprocessor (NFT_PreProcessing.ImagePreprocesser) to binarize the image and attempt to correct blurriness. The parameters have been tuned on a specific set of images and I've noticed that there are some images that it performs poorly on, so use with caution.

After this preprocessing, which is optional, the code uses tesseract and then attempts to read and organize the results.

### Requirements
This requires tesseract version 4 to be installed and available in the path. There are fined tuned libraries specific for the task in OCR/data/.
The rest of the requirements are listed in the setup.py file and will be installed automatically

### Installation

```pip install .```

### Tests

There are a couple of OCR images I've included in tests/test_images/ 

```nosetests tests/test_OCR.py```
