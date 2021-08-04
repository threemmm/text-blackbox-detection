import setuptools
from textdetection import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textdetection",
    version=__version__,
    author="Mohammad Maghsoudimehrabani",
    author_email="maghsoudismtp@gmail.com",
    description="A blackbox detection on text against adversarial attacks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CyberScienceLab/text-blackbox-detection",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: CSL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)