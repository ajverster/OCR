
import re
import regex
import enchant
import Levenshtein as lev
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import tqdm
import re
import pkg_resources
from OCR import Google_OCR_API

def split_ingrdients(block_text):
    """
    Splits the english and french version, if possible
    Assumes you have sometihng like "Ingredients ...... Ingrédeants .... "
    """
    ingr_title = regex.findall('(Ingredients){e<=2}', block_text)
    if len(ingr_title) == 2:
        if block_text.index(ingr_title[0]) > 5:
            return None, None
        i = block_text.index(ingr_title[1])
        return block_text[:i].rstrip(" "), block_text[i:].lstrip(" ")
    return None, None


def process_str(item):
    """
    Remove non-alpha. Strip white space on the ends. Set to lower case
    """
    item = re.sub("[^A-Za-z éèêâ]+","",item)
    item = item.rstrip(" ").lstrip(" ")
    item = item.lower()
    return item


def find_ingredients_lists(response, all_ingredients, mismatches=1):
    """
    Identifies the ingredient list string from the google API response
    """
    switch = False
    results = []
    txt = ""
    for block in response.full_text_annotation.pages[0].blocks:
        block_text = Google_OCR_API.combine_block(block)
        block_text_split = re.split(",|\.|;|:", block_text)
        block_text_split = [process_str(x) for x in block_text_split]
        if any([lev.distance(b,"ingredients") <= mismatches for b in block_text_split]) | ("ingredients" in block_text.lower()):
            if switch:
                results.append(txt)
                txt = ""
            switch = True
        if any([x in block_text.lower() for x in all_ingredients]):
            if switch:
                if len(txt) == 0:
                    txt = block_text
                else:
                    txt = txt + ", " + block_text
        else:
            switch = False
            if len(txt) > 0:
                results.append(txt)
                txt = ""
    if switch:
        # Record the last bit of text
        if len(txt) > 0:
            results.append(txt)
    return results


def preprocess_ingredients(ingredients):
    # gets rid of the stuff after the first period (usually allergy info etc.) if there is one
    step1 = ingredients.dropna().str.lower().str.extract(r'^(.*?)(\.|$)')[0]

    # removes things in parenthesis since they look generally low value
    # TODO: add [], remove space from regex
    step2 = step1.str.replace(r'( \(.*?\))', '')
    step2 = step2.str.replace(r'( \[.*?\])', '')

    # replaces and/or, & with commas (creates two items but might want to remove the second instead)
    # TODO: &/or, '/'
    step3 = step2.str.replace(r'(\s(?:(?:(?:and)|(?:or))\s?/?\s?(?:or)?|&)\s)', ', ')

    # remove 'ingredients:'
    # TODO: remove 'x:' at the beginning
    step4 = step3.str.replace(r'(ingredients:\s?)', '')

    # handle • as separator and period (second case gets filtered by step 1)
    step5 = step4.str.replace(r'( • )', ', ')

    # remove footnote markers *, †, ¹, etc.
    # TODO: * prefix
    step6 = step5.str.replace(r'([^a-z]*,)', ', ')

    # remove "x%" (% may not be there)

    # remove extraneous quotes
    step7 = step6.str.replace('"', ', ')

    # remove lists after ':'

    # remove "contains less than 2% of"
    return step7.fillna('')


def load_ingredient_list(min_length=5, remove_NFT_words=False):
    """
    Loads up a list of (correctly spelled) ingredients from the FLAIME database
    """
    df_ingredients = pd.read_excel(pkg_resources.resource_filename('OCR', 'data/df_ingredients.xlsx'))
    ingredients = preprocess_ingredients(df_ingredients["ingredients"])

    all_ingredients = set()
    for ingr in tqdm.tqdm(ingredients, desc="loading and processing ingredients list"):
        for item in re.split(",|\.|;|:| |\(|\)", ingr):
            # Remove spaces at the stop and end of the word
            item = process_str(item)

            if item in set(stopwords.words('english')):
                continue
            if len(item) < min_length:
                continue
            all_ingredients.add(item)
    all_ingredients.add("ingredients")

    # Remove words that are on the NFT
    if remove_NFT_words:
        words_discard = ["Valeur", "Fat", "Lipides", "Carbohydrates", "Glucides", "Fibre", "fibers", 'Fibres', 'Sugars',
                         'Sucres', 'Protein', "proteins", "Proteines", "Sodium", 'Potassium', "Calcium", "Iron",
                         "Vitamin A", "Vitamin C"]
        words_discard = [process_str(x) for x in words_discard]
        mismatches = 1
        for x in words_discard:
            ingr_found = [ingr for ingr in all_ingredients if lev.distance(x, ingr) <= mismatches]
            for ingr in ingr_found:
                all_ingredients.remove(ingr)
    return all_ingredients


def spell_correct_string(ingr_string, all_ingredients, min_len=5):
    """
    Spell corrects an ingredients string using spell_correct_word() on individual ingredients
    """

    words = [x for x in re.split(",|\.|;|:| |\(|\)", ingr_string) if x != ""]

    # create a map between the split words and their position in the original sentence
    i = 0
    word_map = {}
    for j, w in enumerate(words):
        while ingr_string[i:i + len(w)] != w:
            i += 1
        word_map[j] = i
        i = i + len(w) + 1

    n_corrections = 0
    for j, w in enumerate(words):
        # small words are not included in all_ingredients
        if len(w) < min_len:
            continue

        w_correct = spell_correct_word(w.lower(), all_ingredients)
        if w_correct is None:
            continue
        if w.lower() == w_correct:
            continue
        n_corrections += 1
        ingr_string = ingr_string[:word_map[j]] + w_correct + ingr_string[word_map[j] + len(w):]
        if len(w_correct) != w:
            # Need to ajust all further mapped words
            diff = len(w_correct) - len(w)
            for k in range(j, len(words)):
                word_map[k] = word_map[k] + diff
    return ingr_string, n_corrections


def spell_correct_word(word_oi, all_ingredients_words, n_diff=2):
    """
    Tries to correct word_oi based on the available dictionary (all_ingredients_words). Uses the levevenshtein distance
    """
    words_found = [(x, lev.distance(x, word_oi)) for x in all_ingredients_words if lev.distance(x, word_oi) <= n_diff]
    words_found = sorted(words_found, key=lambda x: x[1])

    if len(words_found) == 1:
        return words_found[0][0]
    if len(words_found) == 0:
        return None
    # Only return if this is the best hit
    if words_found[1][1] > words_found[0][1]:
        return words_found[0][0]
    return None


def check_language(txt):
    """
    Determines whether a given text string is in english or french
    """
    words = split_to_words(txt)
    d_fr = enchant.Dict("fr_CA")
    d_en = enchant.Dict("en_CA")
    words_sub = [x for x in words if (x not in stopwords.words('english')) & (x not in stopwords.words('french'))]
    n_total = len(words_sub)
    n_fr = np.sum([d_fr.check(x) for x in words_sub])
    n_en = np.sum([d_en.check(x) for x in words_sub])
    if (n_fr < n_total/2) & (n_en < n_total/2):
        return "unclear"
    if (n_fr > n_en):
        return "french"
    if (n_fr < n_en):
        return "english"
    return "unclear"


def ocr_all_ingredients(ingr_images_list):
    """
    Main subroutine to OCR ingredients from a list of images
    Calls OCR, extracts ingredients, identifies the language and runs the spell correction
    """
    results_full = []

    all_ingredients = load_ingredient_list()
    # Call the OCR on the full list
    response_elements = Google_OCR_API.ocr_file_set(ingr_images_list)
    for infile in response_elements:
        response = response_elements[infile]
        # Find the ingredients strings from the OCR results
        results = find_ingredients_lists(response, all_ingredients)
        for ingr_text in results:
            # Split each string into english and french versions
            ingr_text_1, ingr_text_2 = split_ingrdients(ingr_text)
            if ingr_text_1 is None:
                results_full.append([infile, ingr_text])
            else:
                results_full.append([infile, ingr_text_1])
                results_full.append([infile, ingr_text_2])
    df_ingr = pd.DataFrame(results_full, columns=["Filename", "ingr_string"])

    # Identify english vs french
    df_ingr["language"] = [check_language(txt) for txt in df_ingr["ingr_string"].values]

    # Do the spell correction
    corrected_text = []
    n_corrections_all = []
    for (i, dat) in df_ingr.iterrows():
        if dat["language"] == "english":
            ingr_string, n_corrections = spell_correct_string(dat["ingr_string"],all_ingredients)
            corrected_text.append(ingr_string)
            n_corrections_all.append(n_corrections)
        else:
            corrected_text.append(dat["ingr_string"])
            n_corrections_all.append(0)

    df_ingr["ingr_string_corrected"] = corrected_text
    df_ingr["n_corrections"] = n_corrections_all
    return df_ingr


def split_to_words(txt):
    """
    Takes an ingredients list as a text string and tries to split it up into a list of individual ingredients
    """
    words = re.split(",|\.|;|:",txt)
    words = [process_str(x) for x in words]
    words = [x for x in words if len(x) > 0]
    words = [x.split(" ") for x in words]
    return [item for sublist in words for item in sublist if item != ""]


def extract_expiry(s):
    """
    Extracts the expiry date from a string
    """
    m = re.search("(20[0-9]{2}[ :\/]{0,1}[A-Z]{2,3}[ :\/]{0,1}[0-9]{2})",s)
    if m:
        return(m.group(1))
    m = re.search("([A-Z]{2,3}[ :\/]{1}[0-9]{2}[ :\/]{1}20[0-9]{2})",s)
    if m:
        return(m.group(1))
    return None

