from OCR import NFT_OCR

import re
import numpy as np
import pandas as pd

from pathlib import Path
import cv2
import copy
from nltk.corpus import stopwords

import pkg_resources


class NFTBoxDetection():
    def __init__(self):
        self.min_line_len=100
        self.line_conection_param=10

        self.load_nft_words()
        self.load_ingredients_list()

    def load_nft_words(self):
        self.nft_words = ["fat", "saturated", "trans", "cholesterol", "sodium", "carbohydrate", "fibre", "sugars",
                         "protein", "calories", "monounsaturated","polyunsaturated","omega-6","omega-3"]
        self.nft_words.extend(["vitamin a","vitamin c","vitamin e", "iron","calcium", "phosphorus","magnesium","zinc"])
        self.nft_words.extend(["nutrition facts", "amount", "daily value"])


    def load_ingredients_list(self):
        """
        Loads up a list of all ingredients
        :return:
        """
        df_ingredients = pd.read_excel(pkg_resources.resource_filename('OCR', 'data/df_ingredients.xlsx'))
        all_ingr_lists = df_ingredients["value"]
        self.ingr_words = set()
        for ingr_list in all_ingr_lists:
            if pd.isnull(ingr_list):
                continue
            for ingr in ingr_list.split(" "):
                word = re.sub("[^A-Za-z]+", "", ingr).lower()
                # Remove stopwords
                if word in set(stopwords.words('english')):
                    continue
                self.ingr_words.add(word)


    def load_image(self, infile):
        self.image = cv2.imread(str(infile))
        self.ocr_image()

    def ocr_image(self):

        self.ocr_nutr = NFT_OCR.OCRWorker(indir_training_files=pkg_resources.resource_filename('OCR', 'data/'),
                                      trained_name="nutrienttraining_int")
        self.ocr_nutr.load_ocr_data(img=self.image)

        self.ocr_ingr = NFT_OCR.OCRWorker(indir_training_files=pkg_resources.resource_filename('OCR', 'data/'),
                                      trained_name="ingredienttraining_int")
        self.ocr_ingr.load_ocr_data(img=self.image)

        self.word_boxes = ocr_to_boxes(self.ocr_nutr.data, min_len=1, trim_down_l=10, trim_down_t=2) + ocr_to_boxes(
                                self.ocr_ingr.data, min_len=1, trim_down_l=10, trim_down_t=2)

    def ingredients_present(self, min_word_length = 3, min_ingredient_words=3):
        # Want to exclude words that were also found in the nutrient words, because there are too many ingredient words and too few nutrient words
        self. ingr_words_found = [x for x in self.ingr_words if (len(
            NFT_OCR.find_string_in_ocr(self.ocr_ingr.data, x)) > 0) &
                                  (len(NFT_OCR.find_string_in_ocr(self.ocr_nutr.data, x)) == 0)]

        self.ingr_words_found = [x for x in self.ingr_words_found if len(x) >= min_word_length]
        self.nutrient_words_found = [x for x in self.nft_words if (len(
            NFT_OCR.find_string_in_ocr(self.ocr_nutr.data, x)) > 0)]

        self.find_item_locs()

        if len(self.ingr_words_found) >= min_ingredient_words:
            return True
        return False

    def find_item_locs(self):
        self.nutrient_items = []
        for n in self.nutrient_words_found:
            found = NFT_OCR.find_string_in_ocr(self.ocr_nutr.data, n)
            if len(found) == 1:  # Otherwise there is a risk this in an ingredient, albeit a small risk
                self.nutrient_items.append(found[0])

        self.ingr_items = []
        for n in self.ingr_words_found:
            found = NFT_OCR.find_string_in_ocr(self.ocr_ingr.data, n)
            if len(found) == 1:  # Otherwise there is a risk this in an ingredient, albeit a small risk
                self.ingr_items.append(found[0])

    def find_vertical_lines(self):

        kernel_length = np.array(self.image).shape[1] // 80
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        img_temp2 = cv2.erode(self.image, verticle_kernel, iterations=3)
        vertical_lines_img = cv2.dilate(img_temp2, verticle_kernel, iterations=3)

        edges = cv2.Canny(vertical_lines_img, 10, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, np.array([]), self.min_line_len, self.line_conection_param)
        # Some of these lines are actually vertical
        lines = [l for l in lines if np.abs(l[0][3]-l[0][1]) > np.abs(l[0][2] - l[0][0])]

        # Eliminate those that touch any of the words
        lines = [l for l in lines if not any([box_line_intersection(l[0], box) for box in self.word_boxes])]

        # If we don't have any lines, assign lines to the edges of the image
        if len(lines) == 0:
            lines = [[0,0,0,self.image.shape[1]],[self.image.shape[0],0,self.image.shape[0],self.image.shape[1]]]
            lines = np.array(lines)
        else:
            lines = np.array(lines)[:, 0, :]
        return lines

    def find_horizontal_lines(self):

        kernel_length = np.array(self.image).shape[1] // 80
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        img_temp2 = cv2.erode(self.image, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

        edges = cv2.Canny(horizontal_lines_img, 10, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, np.array([]), self.min_line_len, self.line_conection_param)
        # Some of these lines are actually horizontal
        lines = [l for l in lines if np.abs(l[0][3]-l[0][1]) < np.abs(l[0][2] - l[0][0])]

        # Eliminate those that touch any of the words
        lines = [l for l in lines if not any([box_line_intersection(l[0], box) for box in self.word_boxes])]
        lines = np.array(lines)[:, 0, :]
        return lines

    def filter_lines_by_ingredients(self, lines_sub, lines_limit):
        names_ingrdients = np.array([self.ocr_ingr.data["text"][i] for i in self.ingr_items])

        if (lines_limit == "top") | (lines_limit == "bottom"):
            pos_ingrdients = np.array([self.ocr_ingr.data["top"][i] + self.ocr_ingr.data["height"][i] / 2 for i in self.ingr_items])
        if (lines_limit == "left") | (lines_limit == "right"):
            pos_ingrdients = np.array([self.ocr_ingr.data["left"][i] + self.ocr_ingr.data["width"][i] / 2 for i in self.ingr_items])


        discard = []
        for i, line in enumerate(lines_sub):
            if lines_limit == "top":
                items_oi = line[1] < pos_ingrdients
            elif lines_limit == "bottom":
                items_oi = line[1] > pos_ingrdients
            elif lines_limit == "left":
                items_oi = line[0] < pos_ingrdients
            elif lines_limit == "right":
                items_oi = line[0] > pos_ingrdients

            names_sub = names_ingrdients[items_oi]
            # Discard lines that contain too many ingredients
            if (np.sum(items_oi) / len(pos_ingrdients) > 0.5) | (
                        any(["ingredients" in x.lower() for x in names_sub])):
                discard.append(i)

        # If they all fail, then this is the wrong side to be looking at
        if len(discard) != len(lines_sub):
            lines_sub = [line for i, line in enumerate(lines_sub) if i not in discard]
        return lines_sub

    def crop_image(self, buffer = 10):
        lines_h = self.find_horizontal_lines()
        lines_v = self.find_vertical_lines()
        lines_horizontal_y_pos = np.array([x[1] for x in lines_h])
        lines_vertical_x_pos = np.array([x[0] for x in lines_v])

        top, bottom, left, right = find_bounding_region_multiple_items(self.ocr_nutr.data, self.nutrient_items)

        lines_h_sub = lines_h[lines_horizontal_y_pos < top + buffer]
        lines_h_sub = self.filter_lines_by_ingredients(lines_h_sub, "top")
        if len(lines_h_sub) == 0:
            line_top = 0
        else:
            line_top = lines_h_sub[np.argmax([np.abs(x[2] - x[0]) for x in lines_h_sub])][1]
        lines_h_sub = lines_h[lines_horizontal_y_pos > bottom - buffer]
        lines_h_sub = self.filter_lines_by_ingredients(lines_h_sub, "bottom")
        if len(lines_h_sub) == 0:
            line_bottom = self.image.shape[0]
        else:
            line_bottom = lines_h_sub[np.argmax([np.abs(x[2] - x[0]) for x in lines_h_sub])][1]

        lines_v_sub = lines_v[lines_vertical_x_pos < left + buffer]
        lines_v_sub = self.filter_lines_by_ingredients(lines_v_sub, "left")
        if len(lines_v_sub) == 0:
            line_left = 0
        else:
            line_left = lines_v_sub[np.argmax([np.abs(x[3] - x[1]) for x in lines_v_sub])][0]
        lines_v_sub = lines_v[lines_vertical_x_pos > right - buffer]
        lines_v_sub = self.filter_lines_by_ingredients(lines_v_sub, "right")
        if len(lines_v_sub) == 0:
            line_right = self.image.shape[1]
        else:
            line_right = lines_v_sub[np.argmax([np.abs(x[3] - x[1]) for x in lines_v_sub])][0]

        image_crop = self.image[line_top:line_bottom, line_left:line_right]
        return image_crop

    def plot_lines(self):
        lines_h = self.find_horizontal_lines()
        lines_v = self.find_vertical_lines()

        image_morph = copy.copy(self.image)
        for line in lines_h:
            x1, y1, x2, y2 = line
            cv2.line(image_morph, (x1, y1), (x2, y2), (220, 20, 20), 2)

        for line in lines_v:
            x1, y1, x2, y2 = line
            cv2.line(image_morph, (x1, y1), (x2, y2), (20, 220, 20), 2)

        return image_morph

    def crop_if_needed(self, infile):
        self.load_image(infile)


        if self.ingredients_present():
            # Abysmally bad images should not be cropped
            if (len(self.nutrient_items) == 0):
                return self.image, False

            return self.crop_image(), True
        else:
            return self.image, False


def find_bounding_region_multiple_items(ocr_data, items_i):
    top = min([ocr_data["top"][i] + ocr_data["height"][i]/2 for i in items_i])
    bottom = max([ocr_data["top"][i] + ocr_data["height"][i]/2 for i in items_i])
    left = min([ocr_data["left"][i] + ocr_data["width"][i]/2 for i in items_i])
    right = max([ocr_data["left"][i] + ocr_data["width"][i]/2 for i in items_i])

    return (int(top), int(bottom), int(left), int(right))



def ocr_to_boxes(ocr_data, min_len=2, trim_down_l=5, trim_down_t=2):
    boxes = []
    i_text_use = [i for i in range(len(ocr_data["text"])) if len(ocr_data["text"][i]) >= min_len]
    for i in i_text_use:
        l, t, w, h = ocr_data["left"][i] + trim_down_l, ocr_data["top"][i] + trim_down_t, ocr_data["width"][
            i] - trim_down_l * 2, ocr_data["height"][i] - trim_down_t * 2
        w = max(w, 1)
        h = max(h, 1)
        boxes.append((l, t, w, h))
    return boxes


def pos_on_line(line, x_new):
    x1, y1, x2, y2 = line
    if x2 == x1:
        return (y1+y2)/2
    m = (y2 - y1) / (x2 - x1)
    y_new = y1 + (x_new - x1) * m
    return y_new


def box_line_intersection(line, box):

    x1, y1, x2, y2 = line
    l, t, w, h = box
    assert x2 >= x1
    ytop = min(y1, y2)
    ybottom = max(y1, y2)
    # If the box is outside of the bounds made by the line
    if (l > x2) | ((l + w) < x1):
        return False
    if ((t + h) < ytop) | (t > ybottom):
        return False

    # Otherwise, we need the top-left of the box above the line and the bottom right below the box
    top_left = (l, t)
    bottom_right = (l + w, t + h)
    if (pos_on_line(line, top_left[0]) > top_left[1]) & (pos_on_line(line, bottom_right[0]) < bottom_right[1]):
        return True
    return False


def draw_image_lines_boxes(infile, boxes, lines):
    image = cv2.imread(infile)

    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (220, 20, 20), 2)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 2)

    return image
