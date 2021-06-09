# Interpreting CNNs for Device Free Wi-Fi Fingerprinting Indoor Localization (IEEE ACCESS 2019)
######  Last update: 6/9/2021
## Introduction:
Implementation of Layer Wise Relevance Propagation (LRP) and Channel Nullfication for Device Free Wi-Fi Fingerprinting Indoor Localization.
For more details and evaluation results, please check out our original [paper](https://ieeexplore.ieee.org/document/8915770 "Title").
## Features:

- LRP: Compute the relevance score of a pre-trainined model under CSI device free indoor localization.
- nullification: Compute the Channel Nullfication with relevance score as the reference
- visualization: plot CSI samples with relevance score
- models: Pre-trainined model (DNN, CNN)
- CSI: Precollected CSI testing set

## Dependencies:
- tensorflow 1.13
- python 3.6
