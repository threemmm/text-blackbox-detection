## Adversarial examples detection on text (BlackBox Attacks)

[![PyPI version](https://badge.fury.io/py/textdetection.svg)](https://badge.fury.io/py/textdetection)

This repo tries to detect black box attacks on text models.
 It doesn't need any models, but a list of strings that 
 an attacker uses to find perturbed versions 
 of sentences that fool the target model. It uses Levenshtein package (ratio) to 
 calculate similarity between different queries.
 

### Requirements
The main requirements are:
- Python 3
- Levenshtein
- Numpy
- Pandas
- Matplotlib
- Scipy
- tqdm

### Usage
Example0 and Example_threshold are provided to show how to use the module. Other examples were used to carry out different experiments.  