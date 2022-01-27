This module contains the code required to do OCR of grocery products. The best code is is NFT_OCR which is used to extract nutrition information from images of NFTs, but there are also modules for doing OCR on Ingredient lists and UPC codes.

### Details

This uses tesseract to extract nutrition information out of images of NFTs. There are a number of preprocessing steps that are worth describing:

1. First, there is the cropper, that will crop the image out of a larger image of the whole package. This is located in NFT_PreProcessing.NFTBoxDetection
2. There is the image unwrapping code (ImageUnwrapping) which is used to unwrap the curvature of images on curved surfaces like soup cans. It does check to see if any modification is required, so it won't make things worse, but it will slow things down a fair bit, so don't use it if it's not necessary.
3. Finally, there is a general purpose preprocessor (NFT_PreProcessing.ImagePreprocesser) to binarize the image and attempt to correct blurriness. The parameters have been tuned on a specific set of images and I've noticed that there are some images that it performs poorly on, so use with caution.

After this preprocessing, which is optional, the code uses tesseract and then attempts to read and organize the results.

# Ingredients

When applied to ingredients lists the method attempts to extract the ingredients lists, split the french and english versions, and applies a spell correction based on a library of known ingredients words

# UPC

Preprocessing attemps to remove the vertical bars of the UPC code before running the OCR. The two smaller numbers to the left and right can be a problem so the code attemps to crop the image and runs tesseract on those subimages.


### Requirements
This requires tesseract version 4 to be installed and available in the path. There are fined tuned libraries specific for the task in OCR/data/.
The rest of the requirements are listed in the setup.py file and will be installed automatically
Modules other than the NFT module, which depend on google's API instead of tesseract, will require you to obtain an API key (a json file) and put its path in the GOOGLE_APPLICATION_CREDENTIALS environmental variable.


### Installation

```pip install .```

### Tests

There are a couple of OCR images I've included in tests/test_images/ 

```nosetests tests/test_OCR.py```


### Usage

# NFTs

```from OCR import NFT_OCR

# Single image
ocr, info, img, flag = NFT_OCR.run_ocr_and_preprocess(infile_nft, unwrap=True, preprocess=True)
df = info.return_df()

# Directory of images
df_full = NFT_OCR.ocr_full_dir(indir, unwrap=True, preprocess=True)
```

# Ingredients

```
from OCR import Ingredients_OCR
from pathlib import Path
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  = path_to_your_api_key_json
ingredients_images = Path(indir).glob("*ING.jpg")
df_ingr = Ingredients_OCR.ocr_all_ingredients(ingredients_images)
```

# UPC codes
```
from OCR import UPC_OCR
from pathlib import Path
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  = path_to_your_api_key_json
upc_images = Path(indir).glob("*ING.jpg")
df_upc, images_all = UPC_OCR.do_full_images(upc_images)
```

# Other (Eg. Front of Package, Expiry dates)

```

from OCR import Google_OCR_API, Ingredients_OCR
from pathlib import Path
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  = path_to_your_api_key_json
infile_list = Path(indir).glob("*FOP.jpg")
results = Google_OCR_API.ocr_file_set(infile_list)
df = Google_OCR_API.get_all_blocks(results)

# This will extract expiry dates from the extracted text
df["expiry_date"] = [Ingredients_OCR.extract_expiry(x) for x in df["text"]]
```

