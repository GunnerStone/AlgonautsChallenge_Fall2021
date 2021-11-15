import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import numpy as np
import torch

#import our custom preprocessing libraries
import RAFTPreprocessing as RAFT
import MFCCPreprocessing as MFCC
import BDCNPreprocessing as BDCN

# Function to process each video in the training data
def process_video(video_path, preprocess_type):
    if preprocess_type == 'RAFT':
        # Get the video name
        video_name = os.path.basename(video_path)
        
        # Process the video using RAFT
        RAFTparser = ArgumentParser()
        RAFTparser.add_argument("--model", help="restore checkpoint", default="RAFT/models/raft-things.pth")
        RAFTparser.add_argument("--iters", type=int, default=12)
        RAFTparser.add_argument("--video", type=str, default=video_path)
        RAFTparser.add_argument("--save", action="store_true", default=True, help="save demo frames")
        RAFTparser.add_argument("--small", action="store_true", help="use small model")
        RAFTparser.add_argument(
            "--mixed_precision", action="store_true", help="use mixed precision"
        )

        args = RAFTparser.parse_args()
        RAFT.inference(args)
    elif preprocess_type == 'MFCC':
        # Get the video name
        video_name = os.path.basename(video_path)
        MFCC.inference(video_path)
    elif preprocess_type == 'BDCN':
        # Get the video name
        video_name = os.path.basename(video_path)
        BDCN.inference(video_name)



def main():
    #process the video using RAFT for testvid.mp4
    process_video("AlgonautsVideos268_All_30fpsmax/0001_0-0-1-6-7-2-8-0-17500167280.mp4", "BDCN")


if __name__ == "__main__":
    main()
