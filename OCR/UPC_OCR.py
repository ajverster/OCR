import sys
import re
import cv2
import tqdm
import numpy as np
import pandas as pd
import pytesseract
from pathlib import Path
from OCR import NFT_OCR, NFT_PreProcessing, Google_OCR_API
import pkg_resources

class UPCWorker():
    """
    This does preprocessing of a UPC image and extracts the releveant text
    """
    def __init__(self):
        self.tess_data = pkg_resources.resource_filename('OCR', 'data/')
        self.trained_file = "upc_int"
        self.left = None
        self.middle = None
        self.right = None

    def prep(self, infile, cutoff=150, method="cutoff", threshold=True, do_lower=True):
        """
        Identifies and removes the UPC bars. Crops to the lower 25% of the image
        """
        img = cv2.imread(str(infile))
        if threshold:
            img_use = self.threshold_image(img, cutoff, method=method)
        else:
            img_use = img

        if do_lower:
            contours, hierarchy = cv2.findContours(img_use, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i in [i for i, c in enumerate(contours) if self.contour_size(c, img_use)]:
                thresh = self.paint_white(img_use, contours, i)

            cut_point = int(img_use.shape[0] * 0.75)
            img_use = img_use[cut_point:, :]
        return img_use

    def run_google_ocr(self, infile, cutoff=150, method="cutoff", threshold=True, do_lower=True):
        """
        Preprocesses and runs the OCR
        """
        assert Path(infile).exists(), "{} not found".format(infile)
        self.img = self.prep(infile, cutoff, method, threshold, do_lower)
        self.results = Google_OCR_API.call_ocr(img=self.img)

    def find_middle(self):
        """
        Find the middle 10 digits
        """

        self.middle = None
        self.middle_box = None
        for i, item in enumerate(self.results.text_annotations[:-1]):
            item_txt = re.sub("[^0-9]+", "", item.description)
            item_txt_next = re.sub("[^0-9]+", "", self.results.text_annotations[i + 1].description)

            if len(item_txt) == 10:
                self.middle = item_txt
                self.middle_box = Google_OCR_API.get_box_from_verticies(item.bounding_poly)
                break
            elif (len(item_txt) == 5) & (len(item_txt_next) == 5):
                self.middle = item_txt + item_txt_next
                self.middle_box = Google_OCR_API.combine_verticies(item.bounding_poly, self.results.text_annotations[i + 1].bounding_poly)
                break

    def find_left_right(self):
        """
        Identifies OCR elements to the left and the right of the self.middle_box
        """
        lefts = []
        rights = []
        for item in self.results.text_annotations:
            b = Google_OCR_API.get_box_from_verticies(item.bounding_poly)
            x_numeric = re.sub("[^0-9]+", "", item.description)
            if len(x_numeric) != 1:
                continue
            if Google_OCR_API.determine_direction(b, self.middle_box) == "left":
                lefts.append((x_numeric, b))
            elif Google_OCR_API.determine_direction(b, self.middle_box) == "right":
                rights.append((x_numeric, b))

        if len(lefts) == 1:
            self.left = lefts[0][0]
        elif len(lefts) > 1:
            # Sometimes the same letter will be duplicated into two different text elements
            if len(set([x[0] for x in lefts])) == 1:
                self.left = lefts[0][0]

        if len(rights) == 1:
            self.right = rights[0][0]
        elif len(rights) > 1:
            # Sometimes the same letter will be duplicated into two different text elements
            if len(set([x[0] for x in rights])) == 1:
                self.right = rights[0][0]

    def crop_left_right(self, direction="left"):
        """
        Crops the image to try and grab the left or right digits
        """
        # Takes 15% from the left or right end
        i = int(self.img.shape[1] * 0.15)
        if direction == "left":
            thresh_sub = self.img[:, :i]
        elif direction == "right":
            thresh_sub = self.img[:, -i:]

        # Crops white space and adds a border, both of which help the OCR
        img_worker = NFT_PreProcessing.ImagePreprocesser(None, thresh_sub)
        img_worker.img = img_worker.convert_to_rgb(img_worker.img)
        img_worker.crop_white()
        img_worker.img = NFT_OCR.add_border(img_worker.img)

        # Runs tesseract in an effort to extract the single value
        results = "".join(pytesseract.image_to_data(img_worker.img, output_type=pytesseract.Output.DICT,
                                                    config='--psm 13 --tessdata-dir {} -l {}'.format(self.tess_data,
                                                                                                     self.trained_file))[
                              "text"])
        results = re.sub("[^0-9]+", "", results)
        # Should be a single digit
        if len(results) != 1:
            results = ""
        return results

    def contour_size(self, c, img):
        """
        Identifies countours that represent vertical bars
        """
        span = c[:, 0, :].max(0) - c[:, 0, :].min(0)
        if (span[0] / img.shape[1] > 0.50) & (span[1] / img.shape[0] > 0.50):
            return False
        if span[1] / img.shape[0] > 0.50:
            return True
        return False

    def find_interior(self, img, contours, i):
        """
        Identifies all points within a contour
        """
        cimg = np.zeros_like(img)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        return pts

    def paint_white(self, img, contours, i):
        """
        Paints a contour and its interior white
        """
        pts = self.find_interior(img, contours, i)
        img[pts] = 255
        return img

    def threshold_image(self, img, cutoff=127, method="cutoff"):
        """
        Thresholds the image
        """
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if method == "adaptive":
            thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 201, 5)
        elif method == "cutoff":
            ret, thresh = cv2.threshold(imgray, cutoff, 255, 0)
        return thresh

    def report_upc(self):
        """
        Returns the extracted UPC code
        """
        if self.middle is None:
            return None
        if self.left is None:
            self.left = "?"
        if self.right is None:
            self.right = "?"
        return self.left + self.middle + self.right

    def find_full(self):
        code_int = re.sub("[^0-9]","", self.results.text_annotations[0].description)
        if len(code_int) == 12:
            self.left = code_int[0]
            self.middle = code_int[1:-1]
            self.right = code_int[-1]
            return True
        return False
    

    def process_full(self, infile, cutoff=150, method="cutoff", threshold=True, do_lower=True):
        """
        Main function that will run the whole algorithm
        """
        self.run_google_ocr(infile, cutoff, method, threshold, do_lower)

        # First, just try and find the full 12 didgets
        first_is_full = self.verify_first_is_full()
        if first_is_full:
            found_full = self.find_full()
            if found_full:
                return self.report_upc()
        # If that didn't work, find the middle 10, then look for the left and right small didget
        self.find_middle()
        if self.middle is None:
            return None
        self.find_left_right()
        if self.left is None:
            self.left = self.crop_left_right("left")
        if self.right is None:
            self.right = self.crop_left_right("right")
        return self.report_upc()

    def verify_first_is_full(self):
        if len(self.results.text_annotations) == 0:
            return False
        item_full = self.results.text_annotations[0]
        b_full = Google_OCR_API.get_box_from_verticies(item_full.bounding_poly)

        first_is_not_full = False
        for item in self.results.text_annotations[1:]:
            b = Google_OCR_API.get_box_from_verticies(item.bounding_poly)
            if not ((b_full[0] <= b[0]) & (b_full[1] <= b[1]) & (b_full[2] >= b[2]) & (b_full[3] >= b[3])):
                first_is_not_full = True
        return not first_is_not_full

def do_full_images(infile_list, cutoffs=[150], method="cutoff", threshold=True, do_lower=True):
    """
    Runs UPC detection on a full list of upc images
    """
    images_all = []
    upc_codes = []
    file_names = []
    for infile in tqdm.tqdm(infile_list):
        w = UPCWorker()

        # Try a few cutoffs
        for c in cutoffs:
            upc_code = w.process_full(infile, c, method, threshold, do_lower)
            if upc_code is not None:
                break
        images_all.append(w.img)
        upc_codes.append(upc_code)
        file_names.append(infile.name)
    df_upc = pd.DataFrame()
    df_upc["upc_codes"] = upc_codes
    df_upc["file_names"] = file_names
    return df_upc, images_all

