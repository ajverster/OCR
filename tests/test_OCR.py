from OCR import NFT_OCR
import cv2

def test_secondaryocr():
    infile = 'tests/test_images/81Xuoyv4YrL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['sugars']['unit'] == "g"
    assert info.results['sugars']['quantity'] == 0


def test_conflicts():
    # fat
    infile = 'tests/test_images/71zCtuO9MTL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    df = info.return_df()
    assert not df.apply(lambda x: "conflict" in x.values, 0).any()

    # fibre
    infile = 'tests/test_images/71ssMZglckL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    df = info.return_df()
    assert not df.apply(lambda x: "conflict" in x.values, 0).any()

    # Carbohydrates
    infile = 'tests/test_images/71FeFrrbcZL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    df = info.return_df()
    assert not df.apply(lambda x: "conflict" in x.values, 0).any()

def test_multiple_hits():
    # Total Fat
    infile = 'tests/test_images/71FeFrrbcZL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['fat']['quantity'] == 3
    assert info.results['fat']['unit'] == "g"

    # Sugar Alcohol
    infile = 'tests/test_images/81wm-NI28tL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['sugars']['quantity'] == 5
    assert info.results['sugars']['unit'] == "g"

    # Total Sugars
    infile = 'tests/test_images/713BciJFzRL._SL1369_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['sugars']['quantity'] == 1
    assert info.results['sugars']['unit'] == "g"


def test_line_removalerrors():
    infile = "tests/test_images/81FOLmXUrwL._SL1500_.jpg"
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['fat']['unit'] == "g"
    assert info.results['fat']['quantity'] == 4.5

    infile = 'tests/test_images/81WNEhRKqTL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['protein']['unit'] == "g"
    assert info.results['protein']['quantity'] == 6


def test_weird_ocr():
    infile = 'tests/test_images/71Ol9JsSDGL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['protein']['unit'] == "g"
    assert info.results['protein']['quantity'] == 2


def test_multiple_fats():
    infile = 'tests/test_images/71zCtuO9MTL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['fat']['unit'] == "g"
    assert info.results['fat']['quantity'] == 0.5

    # Also has fibre spelled weirdly
    assert info.results['fibre']['unit'] == "g"
    assert info.results['fibre']['quantity'] == 2


def test_units_star():
    # fat is g*
    infile = 'tests/test_images/81OmZKGOBYL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['fat']['unit'] == "g"
    assert info.results['fat']['quantity'] == 3

    # fat is g^(cross)
    infile = 'tests/test_images/71mZWBsyeYL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['fat']['unit'] == "g"
    assert info.results['fat']['quantity'] == 4


def test_slash_combined():
    # This one has english / french

    # replace this with something else
    infile = 'tests/test_images/81YZ6LEp14L._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['sodium']['unit'] == "mg"
    assert info.results['sodium']['quantity'] == 20
    assert info.results['protein']['unit'] == "g"
    assert info.results['protein']['quantity'] == 4

    # This one requires a re-crop
    infile = 'tests/test_images/71RDo9Wm6iL._SL1500_.jpg'
    image = cv2.imread(infile)

    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results["protein"]["unit"] == "g"


def test_multiple_sugars():
    # Differentiate between sugars and sugars alcohols
    infile = "tests/test_images/612HV5StaxL._SL1500_.jpg"

    image = cv2.imread(infile)
    ocr, info, tag = NFT_OCR.run_ocr_image(image)
    assert info.results['sugars']['unit'] == "g"
    assert info.results['sugars']['quantity'] == 13


if __name__ == "__main__":
    test_conflicts()
    test_slash_combined()
    test_line_removalerrors()
    test_multiple_hits()
    test_secondaryocr()
    test_multiple_sugars()
    test_weird_ocr()
    test_multiple_fats()
    test_units_star()
