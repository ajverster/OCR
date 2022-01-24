import pytesseract
import OCR.NFT_PreProcessing as NFT_PreProcessing
from OCR.ImageUnwrapping.imageUnwarp import imageUnwarp

import tqdm
import argparse
from collections import defaultdict
import cv2
import re
import copy
import pkg_resources
from pathlib import Path
from PIL import Image, ImageEnhance
import subprocess
import tempfile
import pandas as pd
import numpy as np
import logging
import Levenshtein as lev
import matplotlib.pyplot as plt
from skimage import filters


def dist_between_items(data, i, j):
    """
    Determines the horizontal distance between two items
    :param data: ocr dict from pytesseract
    :param i: index of item 1
    :param j: index of item 2
    :return: distance between the left side of object 2 and the right side of object 1
    """
    if (i == -1) | (j == -1):
        return 0
    if (i == j):
        return 0
    return data['left'][j] - data['left'][i]
    # This can cause problems if the widths are too long
    # return data['left'][j] - (data['left'][i] + data['width'][i])


def contrast_enhance(img):
    return ImageEnhance.Contrast(img).enhance(2.0)


def get_items_on_same_line(data, i, fudge_factor=2, partial_overlap=False, fudge_fraction=0.10):
    """
    Finds OCR items on the same line
    :param data:  ocr dict from pytesseract
    :param i: index of item of interest
    :param fudge_factor: number of pixels to fudge
    :return:
    """
    center = data['top'][i] + data['height'][i] / 2.0

    # Get all items on the same line
    results = []
    for j in range(len(data['top'])):

        # Sometimes we get stupidly small heights, such as 1. Needs a fix
        if data['height'][j] > 5:
            height_use = data['height'][j]
        else:
            height_use = data['height'][i]
        # Alternatively, if the top overlaps the bottom, or the bottom overlaps the top
        if partial_overlap:
            bottom_i = data['top'][i] + data['height'][i]
            bottom_j = data['top'][j] + data['height'][j]
            #if j == 49:
            #    qwe
            if (((bottom_i - data['top'][j] > height_use * fudge_fraction) & (data['top'][i] < bottom_j)) |
                    ((bottom_j - data['top'][i] > height_use * fudge_fraction) & (data['top'][j] < bottom_i))):

                dist = dist_between_items(data, i, j)
                results.append({"i": j, "text": data['text'][j], "dist": dist})
        else:
            if ((center > (data['top'][j] - fudge_factor)) &
                    (center < (data['top'][j] + height_use + fudge_factor))):  # If they are on the same page
                # distance
                dist = dist_between_items(data, i, j)
                results.append({"i": j, "text": data['text'][j], "dist": dist})

    # sort by distance
    results = sorted(results, key=lambda x: x['dist'])
    return results


def find_serving(r):
    """
    Given a LineItem() looks for [number] g
    :param r: LineItem()
    :return:
    """
    j = 1
    while j <= len(r.line_items) - 2:
        if ( (bool(re.search("[0-9]+",r.line_items[j]["text"]))) & (r.line_items[j+1]["text"] in ["g","ml"])):
            return j-1
        j += 1
    return None


class NutrientFinder():

    def find_term(self, data, t, translation_dict={}, method = "one_off"):
        """
        Find term t in your ocr data
        if it can't find t, it looks for the french version of t in the translation_dict
        :param data:
        :param t:
        :param translation_dict:
        :return:
        """
        i = self.text_in_items(data, t, method)

        if i is None:
            # Try the french
            if t in translation_dict:
                t = translation_dict[t]
                i = self.text_in_items(data, t, method)
        return i

    def text_in_items(self, data, text_oi, method = "one_off"):
        """
        Finds ocr items starting or ending with text_oi
        :param data: ocr dict from pytesseract
        :param text_oi: text string we are interested in
        :return: index of the found item
        """
        if method == "start_end":
            found = [i for i, x in enumerate(data['text']) if
                     (bool(re.search("^%s" % (text_oi), x))) | (bool(re.search("%s$" % (text_oi), x)))]
            found = np.array(found)
        elif method == "one_off":
            found = find_string_in_ocr(data, text_oi)
            # Need to worry about vitamin a vs vitamin c
            if "vitamin" in text_oi:
                found = find_string_in_ocr(data, text_oi, split_words=False)
                found = [i for i in found if text_oi[-1] == data['text'][i][-1]]

        elif method == "exact":
            found = find_string_in_ocr(data, text_oi, 0)
        return self.filter_text_for_multiple_hits(data, found, text_oi)

    def filter_text_for_multiple_hits(self, data, found, text_oi):
        """
        Deals with multiple hits
        :param data: ocr dict from pytesseract
        :param found: list of indicies of the hits in data
        :param text_oi: search string
        :return:
        """
        if len(found) == 1:
            return found
        if len(found) > 1:
            # Are they all on one line? Just take the left-most one
            lines = split_into_lines(data)
            if len(set(lines[found])) == 1:
                return [found[0]]
            if text_oi == "fat":
                # need to take all the "total fat"

                total_fat = np.array([i for i in found if data['text'][i - 1] == "total"])
                if len(total_fat) > 0:
                    found = total_fat
                else:
                    # Alternatively, ignore everything with saturated to the left
                    total_fat = np.array([i for i in found if "saturated" not in data['text'][i - 1]])
                    if len(total_fat) > 0:
                        found = total_fat
                return found

            if text_oi == "sugars":
                # remove sugars alcohols from found
                if len(data['text']) <= (max(found) + 1):
                    return []
                non_alcohol = np.array([i for i in found if data['text'][i + 1] != "alcohols"])
                found = non_alcohol
                # If we find total sugars, ignore everything else
                total_sugars = np.array([i for i in found if data['text'][i - 1] == "total"])
                if len(total_sugars) > 0:
                    found = total_sugars
                return found

            if text_oi == "carbohydrate":
                # If we find total carbohydrates, ignore everything else
                total = np.array([i for i in found if data['text'][i - 1] == "total"])
                if len(total) > 0:
                    found = total
                return found

            if text_oi == "fibre":
                # Could be soluble fibre or fibre soluble
                # Checking with in rather thatn == checks for both soluble and insoluble
                found = [i for i in found if (i != 0) & (i != len(data['text']) - 1)]
                fibre = np.array(
                    [i for i in found if ("soluble" not in data['text'][i - 1]) & ("soluble" not in data['text'][i + 1])])
                if len(fibre) > 0:
                    found = fibre
            return found
        else:
            # If we found nothing
            return None


class LineWorker():
    """
    Uses the Hough Transform to find lines in the image
    """
    def __init__(self, img, boxes=None):
        # Confusingly, x (width) is the shape[1] and y (height) is the shape[0]
        self.img = img
        self.boxes = boxes

    def crop_between_lines(self, min_pixel_height=10):
        """
        yield subslices of the image between pairs of lines on the image
        :param min_pixel_height:
        :return:
        """
        # Get horizontal lines, in order
        self.get_lines()
        horizontal_lines = self.filter_lines("horizontal")

        for i in range(len(horizontal_lines) - 1):
            # Make sure these don't just have black in the middle
            if not self.check_line_pairs(horizontal_lines[i], horizontal_lines[i + 1]):
                # [line i], [first or second point] [x or y]
                y1 = horizontal_lines[i][0][1]
                y2 = horizontal_lines[i + 1][1][1]
                if (y2 - y1) >= min_pixel_height: # Lines that are too close together will yield nonesense OCR
                    yield self.img[y1:y2, :, :]

    def remove_lines(self, sensitivity=175):
        """
        Removes the lines you identified with the hough transform
        :param sensitivity:
        :return:
        """
        self.get_lines(sensitivity)
        lines = self.filter_lines()

        for i in range(len(lines) - 1):
            if self.check_line_pairs(lines[i], lines[i + 1]):
                self.fill_region(lines[i], lines[i + 1])
        return Image.fromarray(self.img)

    def put_lines_on_image(self, lines):
        """
        Puts lines onto the image in red so that you can see them
        :param lines: fromself.filter_lines()
        :return: PIL Image
        """
        img = copy.copy(self.img)
        for l in lines:
            cv2.line(img, tuple(l[0]), tuple(l[1]), (255, 0, 0), 2)
        return Image.fromarray(img)

    def get_lines(self, sensitivity=175):
        """
        Populates self.lines with HoughTransform lines
        :param sensitivity: parameter to the HoughTransform. Lower = more lines
        """
        # Edges first
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Finds the lines
        lines_parametric = cv2.HoughLines(edges, 1, np.pi / 180, sensitivity)
        self.lines = []
        if lines_parametric is not None:
            for l in lines_parametric:
                for rho, theta in l:
                    self.lines.append(self.convert_parametric_to_euclidean(rho, theta))
        if self.boxes is not None:
            self.filter_lines_boxes()

    def filter_lines_boxes(self):
        self.lines = [l for l in self.lines if not any([NFT_PreProcessing.box_line_intersection((l[0][0], l[0][1], l[1][0], l[1][1]), box) for box in self.boxes])]


    def threshold_xy(self, xy, m):
        """
        Threshold between 0 and m
        :param xy:
        :param m:
        :return:
        """
        if xy < 0:
            return 0
        if xy > m:
            return m
        return xy

    def convert_parametric_to_euclidean(self, r, theta):
        """
        Converts the parametric coordinates from the hough transform to euclidean coordinates
        :param r:
        :param theta:
        :return:
        """
        large_n = self.img.shape[1] * 10

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + large_n * (-b))
        y1 = int(y0 + large_n * (a))
        x2 = int(x0 - large_n * (-b))
        y2 = int(y0 - large_n * (a))
        return [np.array([self.threshold_xy(x1, self.img.shape[1]), self.threshold_xy(y1, self.img.shape[0])]),
                np.array([self.threshold_xy(x2, self.img.shape[1]), self.threshold_xy(y2, self.img.shape[0])])]

    def filter_lines(self, method="horizontal", f=10):
        """
        Filter lines to be either horizontal or vertical
        :param method: horizontal/vertical
        :param f: ratio of vertical change to horizontal change to be considered more or less veritcal/horizontal
        :return:
        """

        lines_keep = []
        for line in self.lines:
            vertical_change = np.absolute(line[0][1] - line[1][1])  # Difference in y values
            horizontal_change = np.absolute(line[0][0] - line[1][0])  # Difference in x values

            if method == "horizontal":
                if horizontal_change > vertical_change * f:
                    lines_keep.append(line)
            elif method == "vertical":
                if vertical_change > horizontal_change * f:
                    lines_keep.append(line)
            else:
                raise Exception("Must be horizontal or vertical")
        # Sort them from top to bottom of the image
        lines_keep = sorted(lines_keep, key=lambda x: x[0][1])

        return lines_keep

    def get_y_on_line(self, line, x_new):
        """
        Given a position x, use the line formula (y=mx+b) to get the y-value for the x_new
        :param line:
        :param x_new:
        :return:
        """
        line_vec = line[1] - line[0]
        line_slope = line_vec[1] / line_vec[0]

        # line[0][1] = first pt, y-value
        y = line[0][1]
        x = line[0][0]
        y_new = y + line_slope * (x_new - x)
        y_new = y_new.astype(int)
        return y_new

    def check_line_pairs(self, line1, line2, darkness_cutoff=100.0):
        """
        Are these two lines the border of a black line?
        :param line1:
        :param line2:
        :param darkness_cutoff:
        :return:
        """

        x_max = self.img.shape[1]
        # try at both ends
        for x_start, x_end in [(int(x_max / 6), int(x_max / 3)), (int(x_max * 2 / 3), int(x_max * 5 / 6))]:
            vals = []
            for x in range(x_start, x_end):
                y1 = self.get_y_on_line(line1, x) + 1
                y2 = self.get_y_on_line(line2, x) + 1
                vals.extend(self.img[y1:y2, x, :])
            if np.median(vals) < darkness_cutoff:
                return True
        return False

    def fill_region(self, line1, line2):
        """
        Fills region in between two lines
        Used to delete the black lines on the nutrition facts label
        :param line1:
        :param line2:
        :return:
        """
        val_fill = (np.median(self.img[:, :, 0]), np.median(self.img[:, :, 1]), np.median(self.img[:, :, 2]))
        for x in range(0, self.img.shape[1]):
            # Empirically the +1 and +2 are right for covering the black lines
            y1 = self.get_y_on_line(line1, x) + 1
            y2 = self.get_y_on_line(line2, x) + 2
            self.img[y1:y2, x, :] = val_fill


class GroupOfLineItems():
    """
    Creates a list of LineItems from an ocr result
    Load up the data from either create_from_fullocr() or create_from_subocr()
    data on a line is held with a LineItems() class
    each LineItems() is held in self.line_items_list
    """

    def __init__(self, translation_dict, indir_training_files, trained_name):
        self.line_items_list = []

        self.translation_dict = translation_dict
        # Tesseract model files
        self.indir_training_files=indir_training_files
        self.trained_name=trained_name

        self.nf = NutrientFinder()

    def yield_lineitems(self):
        """
        yields items in self.line_items_list
        :return:
        """
        for item in self.line_items_list:
            yield item

    def yield_by_nutr(self,nutr):
        """
        yield the LineItems() associated with a given nutrient
        :param nutr:
        :return:
        """
        line_items_list_sub = [x for x in self.line_items_list if x.nutr == nutr]
        for item in line_items_list_sub:
            yield item

    def create_from_subocr(self, image, nutr_list, ocr_data=None):
        """
        Create your LineItems by using LineWorker() and cropping between each pair of lines
        :param infile:
        :param nutr_list:
        :return:
        """

        if ocr_data is not None:
            boxes = NFT_PreProcessing.ocr_to_boxes(ocr_data, min_len=5, trim_down_l=5, trim_down_t=5)
        else:
            boxes = None
        lines = LineWorker(image, boxes)

        img_crop = list(lines.crop_between_lines())
        ocr = OCRWorker(indir_training_files=self.indir_training_files, trained_name=self.trained_name)
        ocr.load_from_subcrops(img_crop)
        ocr.clean_up_ocr()
        data = ocr.grab_data()
        self.create_from_fullocr(data, nutr_list)

    def create_from_fullocr(self, data, nutr_list):
        """
        Create your LineItems by looking for hits in nutr_list on the full data OCR
        :param data:
        :param nutr_list:
        :return:
        """
        for nutr in nutr_list:
            term_locs = self.nf.find_term(data, nutr, self.translation_dict)

            # Some NFT have the word "cholest" instead of "cholesterol"
            if nutr == "cholesterol":
                if term_locs is None:
                    term_locs = self.nf.find_term(data, "cholest", self.translation_dict)
            if term_locs is None:  # if we can't find the term
                continue
            for i in term_locs:
                if (nutr == "saturated") | (nutr == "trans"):
                    self.line_items_list.append(LineItems(data, i, nutr, partial_overlap=True, fudge_fraction=0.1))
                else:
                    self.line_items_list.append(LineItems(data, i,nutr))
        # Need to handle serving size separately
        li_serving = self.find_serving_size(data)
        for li in li_serving:
            self.line_items_list.append(li)

        #if li_serving is not None:
        #    self.line_items_list.append(li_serving)

    def find_serving_size(self, data):
        """
        Extract the serving size from the results
        :param data:
        :return:
        """
        serving_sizes = []
        # Option1 : per 3/4 cup (31g)
        term_locs = self.nf.find_term(data, "per", {"per":"pour"}, "start_end")
        if term_locs is None: # Sometimes its par instead of per
            term_locs = self.nf.find_term(data, "per", {"per":"par"}, "start_end")
        #if term_locs is None: # Sometimes its "proti
        #    term_locs = self.nf.find_term(data, "per", {"per":"portion de"}, "start_end")
        if term_locs is not None:  # If we can find the term
            # Per can appear on the end of the word copper
            term_locs = [i for i in term_locs if data['text'][i] != "copper"]

            # sometimes there are two of these
            # eg. 6000136119175
            if len(term_locs) >= 1:
                for i in term_locs:
                    if data["text"][i-1] != "amount": # Don't want amount per serving
                        serving_sizes.append(LineItems(data, i, "portion_size"))
        # Option 2: Serving 3/4 cup (29 g)
        term_locs = self.nf.find_term(data, "serving", {"serving":"portion"})
        # There are probably going to be more than 1 serving, "serving 3/4 cup" or "amount per serving"
        if term_locs is not None:
            for i in term_locs:
                if i >= 2:
                    if (data["text"][i - 1] == "per") | (data["text"][i - 2] == "amount"):  # Don't want amount per serving
                        continue
                serving_sizes.append(LineItems(data, i, "portion_size"))
        return serving_sizes

    def identify_nutrients(self):
        pass


class LineItems():
    """
    This class holds a list of items on the same line
    It's meant to hold something like (sugar 5 g 5%)
    Always held in a group by GroupOfLineItems()
    """

    def __init__(self, data, i, nutr = None, lr = "right", partial_overlap=False, fudge_fraction=0.10):
        """
        :param data: ocr dict from pytesseract
        :param i:  index of item of interest
        """
        if len(data['text']) > 0:
            self.line_items = get_items_on_same_line(data, i, partial_overlap=partial_overlap, fudge_fraction=fudge_fraction)
            self.line_items = self.filter_left_right(lr)
            self.line_items = self.split("(\d+\.*\d*)")
            self.line_items = self.split("\/") # TODO: replace this with something else.
            self.join()
        else:
            self.line_items = []
        self.nutr = nutr

    def index_by_text(self, txt, mismatches=1):
        """
        Finds a txt in self.line_items
        :param txt: string of interest
        :return: index of txt in self.line_items
        """
        # Go from right to left
        for i in range(len(self.line_items) - 1, 0 - 1, -1):
            if lev.distance(self.line_items[i]['text'], txt) <= mismatches:
                return i
        return None

    def filter_left_right(self, lr="right"):
        """
        Filters the items so that you only have items to the right of your item of interest
        :param lr:
        :return: subset of self.line_items
        """
        if lr == "right":
            return [x for x in self.line_items if x['dist'] >= 0]
        elif lr == "left":
            return [x for x in self.line_items if x['dist'] <= 0]
        elif lr == "both":
            return self.line_items
        else:
            raise Exception("Invalid lr")

    def split(self, regex_split='(\d+\.*\d*)'):
        """
         If we have any pieces that are combined, eg 25g, then we need to split these
         Subsets self.line_items
        :param regex_split: (\d+\.*\d*) - splits by an int, and optionally a float (carb5g) and (carb1.0g) both work.
        :return: subset of self.line_items
        """
        r_new = []
        for j, item in enumerate(self.line_items):
            item_split = re.split(regex_split, item["text"])
            item_split = [x for x in item_split if x != ""]
            if len(item_split) == 1:
                r_new.append(item)
            elif len(item_split) > 1:
                for k in range(len(item_split)):
                    item_new = {"i": item['i'], "text": item_split[k], "dist": 0}
                    r_new.append(item_new)
        return r_new

    def join(self):
        """
        combines vitamin and a to 'vitamin a'
        :return:
        """
        to_del = []
        for j, item in enumerate(self.line_items):
            if (lev.distance(item['text'], 'vitamin') <= 1) | (lev.distance(item['text'], 'vitamine') <= 1):
                if len(self.line_items) <= j + 1:
                    continue
                if len(self.line_items[j + 1]['text']) == 1:
                    to_del.append(j + 1)
                    self.line_items[j]['text'] = self.line_items[j]['text'] + " " + self.line_items[j + 1]['text']

        self.line_items = [x for i, x in enumerate(self.line_items) if i not in to_del]


class OCRWorker():
    """
    This class runs tesseract and cleans up a bunch of the common issues that appear in the results
    Normal usage is:
    ocrworker = OCRWorker()
    ocrworker.load_from_subcrops() or ocrworker.load_ocr_data()
    ocrworker.clean_up_ocr()
    """

    def __init__(self, infile=None, indir_training_files="tesseract_training/", trained_name="nutrienttraining_allfonts"):
        self.infile = str(infile)
        self.indir_training_files=indir_training_files
        self.trained_name=trained_name
    def load_from_subcrops(self, img_sub_list, offset = 100):
        """
        Given a list of images, this does OCR on each one and concatenates the results together
        :param img_sub_list: list of path to jpg/png imgaes
        :param offset: How much vertical space to put between subcrops
        :return:
        """
        data_full = defaultdict(list)
        for i, img in enumerate(img_sub_list):
            self.load_ocr_data(img=img)
            for key in self.data:
                if key == "top":
                    data_full[key].extend([x + i * offset for x in self.data['top']])
                else:
                    data_full[key].extend(self.data[key])
        self.data = data_full

    def load_ocr_data(self, infile=None, img=None):
        """
        Runs tesseract and returns the OCR dictionary
        :param infile: jpg image file
        :return: ocr dictionary
        """
        if infile is None:
            infile=self.infile

        if img is None:
            self.img = cv2.imread(infile)
        else:
            self.img = img

        if (self.indir_training_files is None) | (self.trained_name is None):
            data = pytesseract.image_to_data(self.img, output_type=pytesseract.Output.DICT)
        else:
            data = pytesseract.image_to_data(self.img, output_type=pytesseract.Output.DICT,
                                             config='--tessdata-dir {} -l {} --psm 11'.format(self.indir_training_files,self.trained_name))
        self.data = data

    def get_resolution(self, infile):
        """
        Uses the unix file command to get the resolution of an image
        :param infile: jpg file
        :return: w,h as integers
        """
        process = subprocess.Popen(['file', infile], stdout=subprocess.PIPE)
        out, err = process.communicate()
        m = re.search(", ([0-9]+) *x *([0-9]+)", out.decode("utf-8"))
        return int(m.group(1)), int(m.group(2))

    def oh_to_zero(self, units=["mg", "g"]):
        """
        Fixes the issue of Omg or Og
        :param data: ocr dict from pytesseract
        :param units: a list of units that we are going to see here
        :return: corrected ocr dict from pytesseract
        """
        for i, item in enumerate(self.data['text']):
            if re.search("^[0-9]*(O|o)(%s)" % ("|".join(units)), item):
                item = re.sub("^([0-9]*)(O|o)(%s)" % ("|".join(units)), r'\1 0\3', 'Og').replace(" ", "")
                self.data['text'][i] = item

    def french_floats_to_english(self):
        """
        Converts instances of 2,5 to 2.5
        :return:
        """
        for i, item in enumerate(self.data['text']):
            if re.search("[0-9]+,[0-9]+", item):
                self.data['text'][i] = item.replace(",",".")

    def correct_spelling(self):
        """
        Corrects some of the spelling mistakes that foolish companies make
        :param data: ocr dict from pytesseract
        :return: corrected ocr dict
        """
        sp_corrections = {"fiber": "fibre", "satures": "sature"}
        for i, item in enumerate(self.data['text']):
            for key in sp_corrections:
                item = item.replace(key, sp_corrections[key])
            self.data['text'][i] = item

    def renmove_objectionable_items(self, remove_items=["", " "]):
        """
        Removes blank items or anything else that is probably a waste of time.
        :param remove_items:
        :return:
        """
        to_remove = []
        for j in range(len(self.data['text'])):
            if (self.data['text'][j] in remove_items):
                to_remove.append(j)
        data = self.remove_list(to_remove)
        return data

    def drop_below_term(self, string_oi):
        """
        Drops results below a given string
        :param string_oi:
        :return:
        """
        i_list = find_string_in_ocr(self.data, string_oi)
        if len(i_list) == 0:
            return None
        for key in self.data:
            i = i_list[0]
            if key == "img":
                continue
            self.data[key] = self.data[key][:i]

    def remove_calories_from(self, groups):
        """
        The string "calories from" causes problems downstream, so I'm removing it
        :param data:
        :param groups: np.array. either from split_into_lines() or data['block_num']
        :return:
        """
        text = np.array(self.data['text'])
        for b in set(groups):
            block_loc = np.where(groups == b)[0]
            for j in block_loc[:-1]:
                if (text[j] == "calories") & (text[j + 1] == "from"):
                    # remove the rest of this block
                    self.data = self.remove_list(list(range(j, np.max(block_loc) + 1)))
                    break

    def remove_list(self, to_remove):
        """
        Removes any items in to_remove from self.data
        :param to_remove:
        :return:
        """
        for key in self.data:
            if key == "img":
                continue
            self.data[key] = [x for i, x in enumerate(self.data[key]) if i not in to_remove]
        return self.data

    def correct_bad_text(self):
        """
        Removes a series of symbols that cause problems downstream
        :return:
        """
        # Accents
        self.data['text'] = [x.replace("é", "e", -1) for x in self.data['text']]
        # Stars
        self.data['text'] = [x.replace("*", "", -1) for x in self.data['text']]
        #  g^(cross) is read as gt or gf
        self.data['text'] = [x.replace("gt", "g", -1) for x in self.data['text']]
        self.data['text'] = [x.replace("gf", "g", -1) for x in self.data['text']]

        # Double or single cross on sugars can cause a problem
        self.data['text'] = [x.replace("sugarstt", "sugars", -1) for x in self.data['text']]
        self.data['text'] = [x.replace("sugarst", "sugars", -1) for x in self.data['text']]
        self.data['text'] = [x.replace("sucrestt", "sucres", -1) for x in self.data['text']]
        self.data['text'] = [x.replace("sucrest", "sucres", -1) for x in self.data['text']]

        # Remove extraneous spaces
        self.data['text'] = [x.lstrip(" ").rstrip(" ") for x in self.data['text']]
        # Remove brackets
        self.data['text'] = [x.replace("(","",-1).replace(")","",-1) for x in self.data['text']]

    def text_to_lowercase(self):
        """
        Changes the text in self.data to be all lower case
        :return:
        """
        self.data['text'] = [x.lower() for x in self.data['text']]

    def clean_up_ocr(self):
        """
        Main function that cleans up an OCR result
        :return:
        """
        self.french_floats_to_english()
        self.text_to_lowercase()
        self.correct_bad_text()

        # Drop the bottom, extraneous information
        for phrase in ['daily values are based', 'values may be higher', "*amount in", "amount in",
                       "*amount in cereal/dans",
                       "amount in cereal/dans", 'teneur de la', 'teneur de le', 'calories a day']:
            self.drop_below_term(phrase)

        lines = split_into_lines(self.data)
        self.remove_calories_from(lines)

        # Correct some of the errors in this
        self.oh_to_zero()
        self.correct_spelling()

        # Split up pieces like "15g" or "fat/lipides" into their own items
        items_to_split = self.find_items_to_split()
        for (text, regex) in items_to_split:
            self.split_ocr_item(text, regex)

        self.renmove_objectionable_items()

    def grab_data(self):
        """
        Returns self.data
        :return:
        """
        return self.data

    def run_ocr(self, infile=None, img=None):
        self.load_ocr_data(infile, img)
        self.clean_up_ocr()

    def resize_and_ocr(self, min_size=1000, scale_factor=2):
        """
        Runs OCR. Resize the image so that we are the proper size for tesseract
        :param infile:
        :param scale_factor:
        :return:
        """
        x, y = self.get_resolution(self.infile)
        if x < min_size:
            outfile = tempfile.NamedTemporaryFile(suffix=".jpg", delete=True)
            cmd = ["convert", self.infile, "-resize", "%ix%i" % (x * scale_factor, y * scale_factor), outfile.name]
            subprocess.call(cmd)
            self.load_ocr_data(outfile.name)
        else:
            self.load_ocr_data()
        self.clean_up_ocr()

    def find_items_to_split(self, regexs_split = ["(\d+\.*\d*)", "\/"]):
        items_to_split = []
        for text_item in self.data['text']:
            for r in regexs_split:
                split = re.split(r, text_item)
                split = [x for x in split if x != ""]
                if len(split) > 1:
                    items_to_split.append((text_item, r))
                    break

        # Make sure you never got the same item in 2 regexes
        # This causes problems if there are two of the same item in the OCR, such as 0%
        #assert len(set([x[0] for x in items_to_split])) == len(items_to_split)
        return items_to_split

    def remove_item_from_ocr(self, j):
        """
        Removes one of the detected words from the OCR
        :param j:
        :return:
        """
        self.data['text'].pop(j)
        self.data['top'].pop(j)
        self.data['height'].pop(j)
        self.data['left'].pop(j)
        self.data['width'].pop(j)
        self.data['conf'].pop(j)

    def split_ocr_item(self, text_oi, regex_oi):
        """
        Split up an OCR text element with a regex
        Splits them into two elements i, j in the ocr.data object
        :param text_oi:
        :param regex_oi:
        :return:
        """
        split = re.split(regex_oi, text_oi)

        j = self.data['text'].index(text_oi)
        t = self.data['top'][j]
        h = self.data['height'][j]
        new_tops = [t for i in range(len(split))]
        new_heights = [h for i in range(len(split))]

        # left. Split up the width based on the length of words
        l = self.data['left'][j]
        w = self.data['width'][j]
        conf = self.data['conf'][j]

        new_widths = [int(len(x) / len(text_oi) * w) for x in split]
        new_lefts = [l]
        for i in range(len(new_widths) - 1):
            new_lefts.append(new_lefts[-1] + new_widths[i])

        # Remove the non-split item
        self.remove_item_from_ocr(j)

        # Insert the split items
        for i in range(len(split)):
            self.data['text'].insert(j, split[i])
            self.data['top'].insert(j, new_tops[i])
            self.data['height'].insert(j, new_heights[i])
            self.data['left'].insert(j, new_lefts[i])
            self.data['width'].insert(j, new_widths[i])
            self.data['conf'].insert(j, conf)
            j += 1

    def merge_ocr_item(self, i, j):
        """
        Merges two OCR word elements
        :param i:
        :param j:
        :return:
        """
        new_text = self.data['text'][i] + " " + self.data['text'][j]
        new_width = self.data['width'][i] + self.data['width'][j]
        self.data['text'][i] = new_text
        self.data['width'][i] = new_width
        self.remove_item_from_ocr(j)

    def merge_vitamin(self):
        """
        Merges items like "vitamin", "a" to "vitamin a"
        :return:
        """
        locs_vitamin = []
        for s in ["vitamin","vitamine"]:
            locs_vitamin.extend(find_string_in_ocr(self.data, s, mismatches=1))
        locs_vitamin = list(set(locs_vitamin))
        locs_vitamin = sorted(locs_vitamin)
        for i in range(len(locs_vitamin)):
            l = locs_vitamin[i]
            if l+1 in locs_vitamin:
                continue
            if (self.data['text'][l+1] == "/") | (len(self.data['text'][l+1]) > 2):
                continue
            self.merge_ocr_item(l, l+1)
            # Merge vitamin b6
            if (self.data["text"][l] == "vitamin b") | (self.data["text"][l] == "vitamine b"):
                if (self.data["text"][l+1] == '6') & (bool(re.search("[0-9]+",self.data["text"][l+2]))):
                    self.merge_ocr_item(l, l + 1)

            # Need to adjust everything over
            for j in range(i+1, len(locs_vitamin)):
                locs_vitamin[j] -= 1

    def merge_split_nutrients(self, nutrients_complete):
        """
        In response to seeing stuff like 's', 'atures'
        :param nutrients_complete:
        :return:
        """
        for i in range(len(self.data['text']) - 1):
            item_joined = self.data['text'][i] + self.data['text'][i + 1]
            if item_joined in nutrients_complete:
                self.merge_ocr_item(i, i+1)
                # The self.data will be reordered, so we need to start again
                return self.merge_split_nutrients(nutrients_complete)
        return None

    def is_american_nft(self):

        self.merge_vitamin()
        # can be any pair of vitamins!

        vitamin_list = ["vitamin a","vitamin c","vitamin d","vitamin b 6", "calcium","iron","thiamin","folate", "pantothenic",
                        "phosphorus", "magnesium", "zinc", "manganese", "niacin"]

        count = 0
        for vit in vitamin_list:
            i = self.data["text"].index(vit) if vit in self.data["text"] else None
            vit_other = [v for v in vitamin_list if v != vit]
            if i is None:
                continue
            line_text = [x['text'] for x in get_items_on_same_line(self.data, i)]
            if any([v in line_text for v in vit_other]):
                count += 1
        if count >= 3: # >=3 means at least 2 complete lines
            return True
        return False

def crop_to_right(data, i, img, border=10, right_pixels=150):
    """
    Crop number of right_pixels to the right
    :param data: ocr dict from pytesseract
    :param i: index of word you want to crop to the right of
    :param img:
    :param border: border to add to the crop
    :param right_pixels: number of pixels we want to crop to the right
    :return:
    """
    y1 = max(data['top'][i] - border, 0)
    y2 = min(data['top'][i] + data['height'][i] + border, img.shape[0])
    x1 = max(data['left'][i] - border, 0)
    x2 = min(data['left'][i] + data['width'][i] + right_pixels + border, img.shape[1])

    img_crop = img[y1:y2, x1:x2, :]
    return img_crop


def crop_and_ocr(data, i, img, indir_training_files, trained_name):
    """
    Crops to the right of word i, run OCR and return the data
    :param data: ocr dict from pytesseract
    :param i:
    :param img:
    :return: ocr dict from pytesseract
    """
    img_crop = crop_to_right(data, i, img)
    ocr = OCRWorker(infile=None,indir_training_files=indir_training_files, trained_name=trained_name)
    ocr.load_ocr_data(img=img_crop)
    ocr.clean_up_ocr()
    return ocr.grab_data()


def find_string_in_ocr(data, string_oi, mismatches = 1, split_words=True, to_lower=True):
    """
    Finds the location of string_oi in data
    Uses Levenshtein distance to allow imperfect string matches
    :param data: ocr dict from pytesseract
    :param string_oi:
    :param mismatches: number of mismatches we can allow
    :return:
    """
    string_oi_split = string_oi.split(" ")
    i_list = []
    for i in range(len(data['text'])):
        if split_words:
            ocr_join = " ".join(data['text'][i:(i + len(string_oi_split))])
            ocr_join = ocr_join.rstrip(" ")
        else:
            ocr_join = data["text"][i]
        if to_lower:
            ocr_join = ocr_join.lower()
            string_oi = string_oi.lower()
        if lev.distance(ocr_join, string_oi) <= mismatches:
            i_list.append(i)
    return i_list


def split_into_lines(data):
    """
    Assigns a line number to each element in data
    Uses get_items_on_same_line()
    :param data: ocr dict from pytesseract
    :return: a list, same length as the data elements, with numbers corresponding to the lines
    """
    seen = set()
    lines = np.ones(len(data['text'])) * -1
    j = 0
    for i in range(len(data['text'])):
        if i in seen:
            continue
        r = get_items_on_same_line(data, i)
        for item in r:
            seen.add(item['i'])
            lines[item['i']] = j
        j += 1
    assert np.sum(lines == -1) == 0
    return lines


def nutr_missing(df, nutr):
    """
    Is the nutr missing from df? Did we fail to find it?
    :param df: From InfoFinder.return_df()
    :param nutr: nutrient of interest
    :return:
    """
    data = df.loc[nutr, :]
    if nutr == "calories":
        if ("quantity" not in data):
            return True
        elif pd.isnull(data["quantity"]):
            return True
    else:
        if ("unit" not in data) | ("quantity" not in data):
            return True
        elif pd.isnull(data["unit"]) | pd.isnull(data["quantity"]):
            return True
        elif data["quantity"] == "conflict":
            return True
    return False


class InfoFinder():
    """
    Works on an OCRWorker to extract nutrient information
    """
    def __init__(self, ocr, indir_training_files, trained_name):

        self.nutrients = ["fat", "saturated", "trans", "cholesterol", "sodium", "carbohydrate", "fibre", "sugars",
                         "protein", "calories"]
        self.vitamins = ["vitamin a","vitamin c","vitamin e", "iron","calcium", "phosphorus","magnesium","zinc", "potassium"]
        self.terms_oi = self.nutrients + self.vitamins

        self.results = {}
        for t in self.terms_oi:
            self.results[t] = {}
        self.results["portion_size"] = {}

        self.translation_dict = {"fat": "lipides",
                                 "saturated": "satures",
                                 "trans": "trans",
                                 "cholesterol": "cholesterol",
                                 "sodium": "sodium",
                                 "carbohydrate": "glucides",
                                 "fibre": "fibres",
                                 "sugars": "sucres",
                                 "protein": "proteines",
                                 "calories": "calories",
                                 "vitamin a": "vitamine a",
                                 "vitamin c": "vitamine c",
                                 "vitamin e": "vitamine e",
                                 "phosphorus": "phosphore",
                                 "magnesium": "magnesium",
                                 "iron": "fer",
                                 "calcium": "calcium",
                                 "zinc":"zinc",
                                 "potassium": "potassium"}

        self.amount_only = ["cholesterol", "protein", "calories", "portion_size"]
        self.dv_only = [] # No longer happens with the new NFT

        self.ocr = ocr

        nutrients_complete = self.nutrients + self.vitamins
        nutrients_complete = nutrients_complete + [self.translation_dict[x] for x in nutrients_complete]
        self.ocr.merge_split_nutrients(nutrients_complete)
        # tesseract trained model
        self.indir_training_files=indir_training_files
        self.trained_name=trained_name
        self.li = GroupOfLineItems(translation_dict=self.translation_dict,
                                   indir_training_files=self.indir_training_files, trained_name=self.trained_name)

    def extract_value(self, ocr_line, j):
        """
        pulls the numeric value (3 in 3g) out of the LineItem()
        :param ocr_line: LineItem()
        :param j: position of the nutrient in ocr_line
        :return: int or float. None is not found.
        """
        if len(ocr_line.line_items) <= j:
            return None
        if re.search("^[0-9]+$", ocr_line.line_items[j]['text']):  # whole numbers eg 45
            return int(ocr_line.line_items[j]['text'])
        if re.search("^[0-9]+\.[0-9]+", ocr_line.line_items[j]['text']):  # decimal numbers eg 0.5
            return float(ocr_line.line_items[j]['text'])
        return None

    def extract_units(self, ocr_line, j):
        """
        pulls the units (g in 3g) out of the LineItem()
        :param ocr_line: LineItem()
        :param j: position of the nutrient in ocr_line
        :return:
        """
        if len(ocr_line.line_items) <= j:
            return None
        if ocr_line.line_items[j]['text'] in ["g", "mg", "ml","ug"]:
            return ocr_line.line_items[j]['text']
        return None

    def add_to_dict(self, nutr, key, val):
        """
        Adds information to the self.results
        If we have both french and english on the same image, then key is already in the dict
        In that case, if the val is the same we are fine, but otherwise we put in "conflict"
        Usually a conflict is indicative of confusion on the part of the algorithm, not french and english being different
        :param nutr: fat, sugars etc.
        :param key: quantity or unit
        :param val: mg,g or 1,5,2.5
        """
        if val is None:
            return None
        if key not in self.results[nutr]:
            self.results[nutr][key] = val
        else:
            if self.results[nutr][key] == val:
                # they are the same, don't need to do anything
                pass
            else:
                # we need to record that there is conflicting information
                self.results[nutr][key] = "conflict"

    def update_dict(self):
        """
        Deals with the amount_only and dv_only cells.
        :return:
        """
        # Updates so that we have some expected values
        for nutr in self.amount_only:
            self.results[nutr]["dv"] = ""
        for nutr in self.dv_only:
            self.results[nutr]["quantity"] = ""
            self.results[nutr]["unit"] = ""

    def extract_units_and_quantities(self, ocr_line, j, nutr):
        """
        Extracts important information from an OCR line
        :param ocr_line:
        :param j: Location of the nutr
        :param nutr: The nutrient in question (eg. fat)
        :return:
        """
        val = self.extract_value(ocr_line, j+1)
        k = j+2
        if val is None:
            val = self.extract_value(ocr_line,j+2)
            k=j+3
        if nutr in self.amount_only:
            self.add_to_dict(nutr, "quantity", val)
            unit = self.extract_units( ocr_line, k)
            if nutr == "calories":
                self.add_to_dict(nutr, "unit", "kcal")
            else:
                self.add_to_dict(nutr, "unit", unit)
        elif nutr in self.dv_only:
            self.add_to_dict(nutr, "dv", val)
        else:
            # Decide if this is an amount or a dv
            unit = self.extract_units( ocr_line, k)
            if unit is None:
                done = False

                if k < len(ocr_line.line_items):
                    if ocr_line.line_items[k]["text"] == "%":
                        self.add_to_dict(nutr, "dv", val)
                        done = True
                # glucides 6 7% then the first one is val, second one is percent dv
                if not done:
                    if (k + 1) < len(ocr_line.line_items):
                        if ocr_line.line_items[k+1]["text"] == "%":
                            self.add_to_dict(nutr, "quantity", val)
                            self.add_to_dict(nutr, "dv", ocr_line.line_items[k]["text"])

            else:
                self.add_to_dict(nutr, "quantity", val)
                self.add_to_dict(nutr, "unit", unit)
                # Now find the dv as the number number
                val = self.extract_value(ocr_line, k + 1)
                self.add_to_dict(nutr, "dv", val)

    def crop_next_to(self, ocr_line, j):
        """
        Creates a new LineItems by cropping next to j
        :param ocr_line: LineItem
        :param j:  position of nutrient in ocr_line
        :return: LineItem of the crop
        """
        data_sub = crop_and_ocr(self.ocr.data, ocr_line.line_items[j]['i'], self.ocr.img,
                                indir_training_files=self.indir_training_files, trained_name=self.trained_name)
        return LineItems(data_sub, 0)

    def reocr(self,ocr_line, j, nutr):
        """
        Checks if we need to reOCR and if so, uses crop_next_to()
        :param ocr_line: LineItem
        :param j: position of nutrient in ocr_line
        :param nutr: nutrient in question
        """
        # Conditions for re-OCR
        if ("quantity" not in self.results[nutr]) | (("unit" not in self.results[nutr]) & (nutr != "calories")):
            r_new = self.crop_next_to( ocr_line, j )
            # j can't be just zero if we have protein/proteines
            if nutr == "portion_size":
                j = 0
            else:
                j = self.find_nutrient(r_new, nutr, self.translation_dict)
                if j is not None:
                    self.extract_units_and_quantities(r_new, j, nutr)

    def extract_all_info(self, method="full"):
        """
        Main function. Extracts nutrition information from self.ocr
        :param method: full (ocr the whole image) or sub (ocr between pairs of lines
        """

        if method == "full":
            self.li.create_from_fullocr(self.ocr.data, self.terms_oi)
        elif method == "sub":
            self.li.create_from_subocr(self.ocr.img, self.terms_oi, self.ocr.data)
        else:
            raise Exception("Invalid method")

        # Show everything
        for r in self.li.yield_lineitems():
            j = self.find_nutrient(r, r.nutr, self.translation_dict)
            logging.debug(r.nutr)
            logging.debug(j)
            logging.debug([x['text'] for x in r.line_items])

        for nutr in self.terms_oi:
            for r in self.li.yield_by_nutr(nutr):
                j = self.find_nutrient(r, nutr, self.translation_dict)
                if nutr == "cholesterol":
                    if j is None:
                        j = self.find_nutrient(r, "cholest", self.translation_dict)
                if j is None:
                    continue
                self.extract_units_and_quantities(r, j, nutr)
                if method == "full":
                    self.reocr(r,j,nutr)

        # Handle the portion size differently
        for r in self.li.yield_by_nutr("portion_size"):
            j = find_serving(r)
            if j is None:
                continue
            self.extract_units_and_quantities(r, j, "portion_size")
            # TODO: need to reOCR

        # Fill in the
        self.update_dict()

    def return_df(self):
        """
        Results the results as a DataFrame
        :return:
        """
        df_r = pd.DataFrame(self.results, index=["quantity","unit","dv"]).T
        return df_r

    def return_dict(self):
        """
        Returns the results DataFrame converted to a dictionary
        :return:
        """
        d = pd.DataFrame(self.results).to_dict()
        # Change np.nan to None
        for key in d:
            for key2 in d[key]:
                if pd.isnull(d[key][key2]):
                    d[key][key2] = None
        return d

    def find_nutrient(self, r, t, terms_dict):
        """
        Given a LineItem() finds the nutrient, finds the nutrient of interest
        :param r: LineItem()
        :param t: nutrient name
        :param terms_dict: eng->fr translation dictionary
        :return:
        """
        # Need to deal with eng / french
        # Find the right-most version of the nutrient
        # eg. Carbohydate / Glucide <- find Glucide

        j = r.index_by_text(t)
        k = None
        if (t in terms_dict):
            k = r.index_by_text(terms_dict[t])
        if j is None:  # If you can't find the first term, switch over
            j = k
        # assuming they are in order
        if k is not None:
            if k > j:
                j = k
        return j

    def determine_absent_ingredients(self):
        loc = find_string_in_ocr(self.ocr.data, "not a significant source", mismatches=1, to_lower=True)

        absent_str = "absent"

        # Handle not finding this string
        if len(loc) == 0:
            loc_use = np.inf
        else:
            assert len(loc) == 1
            loc_use = loc[0]

        for nutr in self.results:
            if (len(self.results[nutr]) == 0) | all([self.results[nutr][x] == "" for x in self.results[nutr]]):
                hits = self.li.nf.find_term(self.ocr.data, nutr, translation_dict=self.translation_dict, method="one_off")
                if hits is None:
                    self.results[nutr]["quantity"] = absent_str
                    self.results[nutr]["unit"] = absent_str
                    self.results[nutr]["dv"] = absent_str
                elif all([x > loc_use for x in hits]):
                    self.results[nutr]["quantity"] = absent_str
                    self.results[nutr]["unit"] = absent_str
                    self.results[nutr]["dv"] = absent_str

        if self.new_nft():
            self.results["carbohydrate"]["dv"] = absent_str
        else:
            self.results["sugars"]["dv"] = absent_str
            # nutrients do not have amounts
            for nutr in self.vitamins:
                self.results[nutr]["quantity"] = absent_str
                self.results[nutr]["unit"] = absent_str

        # Set the DV missing to absent
        for nutr in self.amount_only:
            self.results[nutr]["dv"] = absent_str

    def new_nft(self):

        loc_en = find_string_in_ocr(self.ocr.data, "5 % or less is a little 15 % or more is a lot", mismatches=4,
                                     to_lower=True)

        loc_fr = find_string_in_ocr(self.ocr.data, "5 % ou moins cest peu 15 % ou plus cest beaucoup", mismatches=4,
                                     to_lower=True)
        if (len(loc_en) > 0) | (len(loc_fr) > 0):
            return True
        # Check if we have vitamin mg amounts
        for nutr in ['potassium','iron','calcium']:
            if nutr in self.results:
                if 'unit' in self.results[nutr]:
                    if self.results[nutr]['unit'] == "mg":
                        return True
        return False


def merge_results(info1, info2, test_conflicts=False):
    """
    If we have 1 result (info1) with NaNs in them, this fills in the missing values from info2
    Generally, test_conflicts should be False if this is a subcrop method, but True if this is a second OCR
    :param info1: InfoFinder()
    :param info2: InfoFinder()
    :return: updated info1
    """
    if info1 is None:
        return info2
    if info2 is None:
        return info1

    keys_oi = ["quantity","unit","dv"]
    for nutr in info1.terms_oi:
        # If we don't have any NaNs, this is fine
        if (not all([key in info1.results[nutr] for key in keys_oi])) | (test_conflicts):
            # Only take info2.results[nutr] if it is better
            # Or record a conflit if we are testing conflicts
            for key in info2.results[nutr]:
                if key not in info1.results[nutr]:
                    info1.results[nutr][key] = info2.results[nutr][key]
                else:
                    if test_conflicts:
                        if info2.results[nutr][key] != info1.results[nutr][key]:
                            info1.results[nutr][key] = "conflict"
    return info1


def preprocess_full(infile, target_width=1000, mode="binary", thresh_val=150, kernel_size=1, slow_mode=False, unwrap=True):
    flag_composite = []

    if not Path(infile).exists():
        return None

    if not slow_mode:
        # First, crop out the NFT
        box_detector = NFT_PreProcessing.NFTBoxDetection()
        image_arr, flag_crop = box_detector.crop_if_needed(infile)

        # Second, unwarp
        flag = -1
        unwarp_stats = None
        if unwrap:
            image_arr, flag, unwarp_stats = imageUnwarp(image=image_arr)
    else:
        image_arr = cv2.imread(infile)

    # Third, pre-process
    img_worker = ImagePreprocesser(None, image_arr)
    img_worker.crop_white()
    img_worker.resize(target_width)
    img_worker.threshold(mode=mode, cutoff=thresh_val)
    img_worker.close(kernel_size=kernel_size)

    # Composite_flag
    if flag_crop:
        flag_composite.append("crop")
    if flag == 2:
        flag_composite.append("unwarp")
    if flag == 100:
        flag_composite.append("unwrapcrash")

    return img_worker.img, flag_composite, unwarp_stats


def run_ocr_and_preprocess(infile, target_width=1000, mode="binary", thresh_val=150, kernel_size=1, slow_mode=False, unwrap=True,
            trained_dir=pkg_resources.resource_filename('OCR', 'data/'),
            trained_name='nutrienttraining_int'):
    """
    Main function. Uses OCRWorker to run the OCR and InfoFinder to extract the nutrient information
    :param infile:
    :return:
    """
    if not Path(infile).exists():
        return None, None, None, "missing"

    img, flag_preprocess, unwarp_stats = preprocess_full(infile, target_width=target_width, mode=mode, thresh_val=thresh_val,kernel_size=kernel_size, slow_mode=slow_mode, unwrap=unwrap)
    ocr, info, flag_ocr = run_ocr_image(img, trained_dir=trained_dir, trained_name=trained_name)

    return ocr, info, img, flag_preprocess + flag_ocr


def run_ocr_image(img, trained_dir=pkg_resources.resource_filename('OCR', 'data/'),
            trained_name='nutrienttraining_int'):
    ocr = OCRWorker(indir_training_files=trained_dir, trained_name=trained_name)
    ocr.run_ocr(img=img)

    if ocr.is_american_nft():
        return None, None, ["american"]

    info = InfoFinder(ocr, indir_training_files=trained_dir, trained_name=trained_name)
    info.extract_all_info("full")
    df = info.return_df()
    if df.apply(lambda x: any(pd.isnull(x)), 0).any():
        info_sub = InfoFinder(ocr, indir_training_files=trained_dir, trained_name=trained_name)
        info_sub.extract_all_info("sub")
        info = merge_results(info, info_sub)
    info.determine_absent_ingredients()
    return ocr, info, ["fine"]


def add_border(img):
    """
    Adds a white border to an image
    :param img:
    :return:
    """
    bordersize = 10
    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    return border



class ImagePreprocesser():
    """
    Class that performs pre-processing on an image to prepare it for OCR
    """
    def __init__(self, infile=None, img=None):
        # Load the image
        if infile is not None:
            self.img = cv2.imread(str(infile))
        else:
            assert img is not None
            self.img=img

    def process(self):
        """
        Runs all the pre-processing steps
        :return:
        """
        self.crop_white()
        self.resize()
        self.close()
        self.threshold()
        self.close()

    def resize(self, target_width = 400):
        """
        Resize an image to target a width of 2,000 pixels
        :param target_width:
        :return:
        """
        if True:
            scale_factor = target_width / self.img.shape[1]
            width = int(self.img.shape[1] * scale_factor)
            height = int(self.img.shape[0] * scale_factor)
            dim = (width, height)
            self.img = cv2.resize(self.img, dim)

    def convert_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def convert_to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def threshold(self, mode = "tozero", cutoff=None, invert=False, block_size=25, C=25):
        """
        Threshold an image
        :param cutoff:
        :param binary:
        :return:
        """
        grayimg = self.convert_to_gray(self.img)
        if invert:
            grayimg = 255 - grayimg
        if cutoff is None:
            cutoff = filters.threshold_otsu(grayimg) / 2
        if mode == "binary":
            ret2, th = cv2.threshold(grayimg, cutoff, 255, cv2.THRESH_BINARY)
        elif mode == "adaptive":
            th = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        elif mode == "tozero":
            ret2, th= cv2.threshold(grayimg, cutoff, 255, cv2.THRESH_TOZERO)
        if invert:
            th = 255-th
        self.img = self.convert_to_rgb(th)

    def close(self, kernel_size=2):
        """
        runs morphologyEx
        :param kernel_size:
        :return:
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)

    def write_img(self):
        """
        Writes an image to a tempfile
        :return:
        """
        outfile = tempfile.NamedTemporaryFile(suffix=".jpg")
        cv2.imwrite(outfile.name, self.img)
        return outfile

    def crop_white(self):
        """
        Crops white border at the edge of the image
        :return:
        """
        # Left to right
        cutoff_l = 0
        for i in range(self.img.shape[0]):
            if self.img[i, :, :].min() < 250:
                cutoff_l = i
                break

        # right to left
        cutoff_r = 0
        for i in range(self.img.shape[0], 0, -1):
            if self.img[(i - 1), :, :].min() < 250:
                cutoff_r = i
                break

        cutoff_t = 0
        for i in range(self.img.shape[1]):
            if self.img[:, i, :].min() < 250:
                cutoff_t = i
                break

        cutoff_b = 0
        for i in range(self.img.shape[1], 0, -1):
            if self.img[:, (i - 1), :].min() < 250:
                cutoff_b = i
                break

        self.img = self.img[cutoff_l:cutoff_r, cutoff_t:cutoff_b, :]


def debugging_show_crops(infile, indir_training_files, trained_name):
    """
    Shows the crops you get with a crop_between_lines
    :param infile:
    :param indir_training_files:
    :param trained_name:
    :return:
    """
    img = cv2.imread(str(infile))
    lines = LineWorker(img)
    img_crop = list(lines.crop_between_lines())
    for i in range(len(img_crop)):
        img = img_crop[i]
        img = add_border(img)

        grayimg = cv2.cvtColor(255 - img, cv2.COLOR_BGR2GRAY)
        ret2, img = cv2.threshold(grayimg, 25, 255, cv2.THRESH_BINARY)
        img = 255 - img

        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        ocr = OCRWorker(indir_training_files=indir_training_files, trained_name=trained_name)
        ocr.load_from_subcrops([img])
        ocr.clean_up_ocr()
        data = ocr.grab_data()
        print(data['text'])

        plt.imshow(img)
        plt.show()
    return img_crop


def ocr_full_dir(indir, unwrap=True):
    for infile in Path(indir).glob("*.jpg"):
        ocr, info, img, flag = run_ocr_and_preprocess(infile, unwrap=unwrap)
        df = info.return_df()
        df["infile"] = infile
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', default='single_image', help='test means to test a single image. full means to go through the entire set of images in the database and label them')
    parser.add_argument('-i','--infile', help='path to the image of a nutrition facts table that you want to extract')
    parser.add_argument('-d','--indir', help='path to the directory of nutrition facts table images you want to extract')
    parser.add_argument('-o','--outfile', help='path to the outfile you wish to write')
    parser.add_argument('-w','--unwrap', default=False, action="store_true", help='Whether your NFTs include those possibly on a cylinder that need to be unwraped. Slows the whole thing down.')

    args = parser.parse_args()

    assert int(re.search('tesseract ([0-9]{1})\.', subprocess.check_output(['tesseract', '--version']).decode('utf-8')).group(1)) >= 4, 'Minimum tesseract version is 4.0'

    if args.task == "single_image":
        ocr, info, img, flag = run_ocr_and_preprocess(args.infile, unwrap=args.unwrap)
        df = info.return_df()
        df.to_excel(args.outfile)
    elif args.task == "full":
        df_full = ocr_full_dir(args.indir, unwrap=args.unwrap)
        df_full.to_excel(args.outfile)
    else:
        raise Exception("Invalid task")

