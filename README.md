# Talking-head-Audio-Visual-Matching

This repository contains the code developed by TensorFlow for implementing (including the preprossing-input pipeline and training/evaluation) the following paper ![3D Convolutional Neural Networks for Cross Audio-Visual Matching Recognition](http://ieeexplore.ieee.org/document/8063416/). The training/evluation parts is strongly inspired by the official project of https://github.com/astorfi/lip-reading-deeplearning, so you can also reference this official project for more details.

![im1](readme_images/1.gif)![im2](readme_images/2.gif)![im3](readme_images/3.gif)

# Objective
This project aims to protect face authentication system from the attack of the synthetic video by detecting the synchronization of video and audio of a talking head by 3D CNNs based on *Audio-visual recognition* (AVR).

# Dataset
This work use the ![Lip Reading Datasets](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/) to train and evluate the algorithm. Since the dataset is based on the video clips of the English news, the models can only be used for the English corpus. 


# Code Implementation
The code includes:
1) input pipline: extracting the audio features (MFCC) and the video clips (image sequences) to compose the positive and negative training samples to train the 3D CNNs;

2) lip tracking: extraing the lip ROI in the video clips considered as visaul features corresponding to the audio features 

3) training processing: training the *coupled 3D convolutional neural network* with the positive and negative pairs of the audio/visual features to recognize the audio-visual matching

# Conclusion
The model for recognizing the audio-visual matching is severely depend on the corpus, e.g. the language used in the corus, the accent/style of the speech and the phrases used in the corpus etc, so that the capacity of the model trained on LRW is very limited to apply it on the non-English context, or even on the other English-based videos but not appeared in the LRW in which the performance is very poor.       

# References
    @article{torfi20173d,
      title={3d convolutional neural networks for cross audio-visual matching recognition},
      author={Torfi, Amirsina and Iranmanesh, Seyed Mehdi and Nasrabadi, Nasser and Dawson, Jeremy},
      journal={IEEE Access},
      volume={5},
      pages={22081--22091},
      year={2017},
      publisher={IEEE}
    }

project site: https://github.com/astorfi/lip-reading-deeplearning

