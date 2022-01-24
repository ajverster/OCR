import setuptools

__version__ = '0.1.0'
__author__ = ['Adrian Verster']
__email__ = 'adrian.verster@hc-sc.gc.ca'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="OCR",
    install_requires=['numba==0.54.0',
                      'tqdm',
                      'opencv-python==4.5.5.62',
                      'numpy==1.20.3',
                      'pandas==1.3.5',
                      'pytesseract==0.3.8',
                      'pathlib==1.0',
                      'pillow==9.0',
                      'sympy==1.9',
                      'scipy==1.7',
                      'matplotlib==3.5',
                      'python-levenshtein==0.12.2',
                      'scikit-image==0.19',
                      ],
    python_requires='>3.6',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ajverster/OCR/tree/main",
    #package_dir={"": "src"},
    packages=["OCR","OCR.ImageUnwrapping"],
    package_data={'': ['data/*']},
    include_package_data=True,
    version=__version__,
    author=__author__,
    author_email=__email__,
)
