## Adversarial examples detection on text (BlackBox Attacks)

[![PyPI version](https://badge.fury.io/py/textdetection.svg)](https://badge.fury.io/py/textdetection)
[![GitHub license](https://img.shields.io/github/license/threemmm/text-blackbox-detection)](https://github.com/threemmm/text-blackbox-detection/blob/master/LICENSE)

This repo tries to detect black box attacks on text models.
 It doesn't need any models, but a list of strings that 
 an attacker uses to find a perturbed versions 
 of sentences that fool the target model. It uses fuzzywuzzy package to 
 calculate similarity between different sentences.
 

### Requirements
The main requirements are:
- Python 3
- fuzzywuzzy
