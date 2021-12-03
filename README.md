# The Algonauts Project 2021
## How the Human Brain Makes Sense of a World in Motion

<p align="center">
  <img src="doc/challenge_overview.png" />
  <br><span>(source: http://algonauts.csail.mit.edu/challenge.html)</span>
</p>

Code for the Algonauts Project 2021 (http://algonauts.csail.mit.edu/). The goal was to predict brain responses recorded while participants viewed short video clips of everyday events. The project had two pars:
- Mini Track: predict brain responses in specific regions of interest (ROIs) of the brain that are known to play a key role in visual perception (V1, V2, V3, V4, LOC, EBA, FFA, STS, PPA)
- Full Track: predict brain responses across the whole brain (for the provided set of reliable voxels)

This solution has not been submitted for official testing on the Algonauts website.

## Description
Data preprocessing: 
- Remove all videos that don't contain audio
- Remove all videos that don't contain at least 75 frames
- Videos resampled to 75 Frames and resided to 32*32 pixels
- Preprocessed Inputs: BDCN edge detector, RAFT motion energy model, MFCC audio encoding

Training:
- Train a regression model for each preprocessed input
- 80/20 split for training and validation
- Loss function: (1-Pearson's Correlation Coefficient)<sup>3</sup>

Models:
- RAW Model: densenet121 backbone; pretrained: False; optimizer: Adam; LR Scheduler Patience: 2
- RAFT Model: cspresnext50 backbone; pretrained: True; optimizer: Adam; LR Scheduler Patience: 2
- BDCN Model: densenet121 backbone; pretrained: False; optimizer: Adam; LR Scheduler Patience: 2
- MFCC Model: densenet121 backbone; pretrained: False; optimizer: Adam; LR Scheduler Patience: 2

![alt text](doc/model_overview.png)

## Report

Report: [doc/tbd.pdf](/doc/tbd.pdf)<br>

## Results
| ROIs 	| RAW 	| RAFT 	| BDCN 	| MFCC 	| Ensemble 	|
|---	|---	|---	|---	|---	|---	|
| V1 	| 0.0676 	| 0.0476 	| <b><u>0.1447</u></b> 	| 0.0319 	| 0.0134 	|
| V2 	| 0.0941 	| 0.0432 	| <b>0.1231</b> 	| 0.0069 	| 0.0208 	|
| V3 	| 0.0445 	| 0.0394 	| <b>0.1320</b>	| 0.0244 	| 0.0100 	|
| V4 	| 0.0366 	| 0.0267 	| <b>0.0958</b>	| 0.0554 	| -0.0406 	|
| EBA 	| 0.1058 	| 0.0752 	| <b>0.1568</b> 	| 0.0345 	| 0.0474 	|
| FFA 	| 0.0383 	| 0.0222 	| <b>0.0704</b> 	| -0.0253 	| -0.0254 	|
| LOC 	| 0.0780 	| 0.0845 	| <b>0.1788</b> 	| 0.0265 	| -0.0231 	|
| PPA 	| 0.0398 	| 0.0666 	| <b>0.0748</b> 	| 0.0226 	| -0.0158 	|
| STS 	| 0.0286 	| 0.0610 	| <b>0.0647</b> 	| 0.0070 	| -0.0399 	|


## TODO:
- try different spatial and temporal resolutions to see which of the two is more important for this task
- get a score higher than the Alexnet baseline
- submit to Algonauts for testing
