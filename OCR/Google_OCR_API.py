
from google.cloud import vision
import io
import cv2
import os
import pandas as pd

def call_ocr(path = None, img = None):
    """
    Calls the google OCR API from either an image file (path) or image array
    """
    assert sum([path is None, img is None]) == 1, "Must provide only 1 of path and img"
    assert "GOOGLE_APPLICATION_CREDENTIALS" in os.environ, "Must set the GOOGLE_APPLICATION_CREDENTIALS environmental variable"

    if path is not None:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()
    else:
        content = cv2.imencode('.jpg', img)[1].tostring()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response


def combine_block(blocks):
    """
    Combines the text in a given block into a single string
    """
    full_text = []
    for para in blocks.paragraphs:
        full_text = full_text + combine_text(para.words)
    return " ".join(full_text)


def get_blocks(response):
    """
    Get the blocks of text as a list from a google OCR API
    """
    text = []
    if len(response.full_text_annotation.pages) == 0:
        return text
    for block in response.full_text_annotation.pages[0].blocks:
        text.append(combine_block(block))
    return text


def get_text_annotations(response):
    """
    Get the text annotations as a list from a google OCR API
    """
    results = []
    for item in response.text_annotations:
        results.append(item.description)
    return results


def ocr_file_set(file_list):
    """
    Runs the google OCR api on a list of files
    Assumes your infiles are pathlib objects
    """
    results = {}
    for infile in file_list:
        response = call_ocr(infile)
        results[infile.name] = response
    return results

def get_all_blocks(results):
    """
    Takes the output of ocr_file_set() and identifies all blocks of text and returns them in a DataFrame
    """
    results_full = []
    for key in results:
        response = results[key]
        text = get_blocks(response)
        for text_part in text:
            results_full.append([key, text_part])

    df = pd.DataFrame(results_full, columns=["file", "text"])
    return df


def get_box_from_verticies(bounding_box):
    """
    Identifies the verticies from a bounding box
    """
    x_list = [v.x for v in bounding_box.bounding_poly.vertices]
    y_list = [v.y for v in bounding_box.bounding_poly.vertices]
    return (min(x_list), min(y_list), max(x_list), max(y_list))


def combine_verticies(bbox1 = None, bbox2 = None, vertex1 = None, vertex2 = None):
    """
    Combines either two verticies or two bounding boxes
    """
    if (bbox1 is not None) & (bbox1 is not None):
        x_list = [v.x for v in bbox1.bounding_poly.vertices] + [v.x for v in bbox2.bounding_poly.vertices]
        y_list = [v.y for v in bbox1.bounding_poly.vertices] + [v.x for v in bbox2.bounding_poly.vertices]
    else:
        x_list = [vertex1[0], vertex1[2], vertex2[0], vertex2[2]]
        y_list = [vertex1[1], vertex1[3], vertex2[1], vertex2[3]]

    return (min(x_list), min(y_list), max(x_list), max(y_list))

def determine_direction(v1, v2):
    """
    Given two sets of verticies, this funciton identifies whether v1 is left or right of v2
    """
    if (v1[2] < v2[0]):
        return "left"
    if (v2[2] < v1[0]):
        return "right"
    return "overlap"

def crop_image(img, vertex):
    """
    Crops an image from a list of verticies
    """
    return img[vertex[1]:vertex[3],vertex[0]:vertex[2]]

def combine_text(words):
    """
    Combines the individual symbols in the object
    """
    full_text = []
    for i in range(len(words)):
        full_text.append("".join([x.text for x in words[i].symbols]))
    return full_text
