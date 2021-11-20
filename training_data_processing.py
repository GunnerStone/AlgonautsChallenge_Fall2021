import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict

from os import listdir
from os.path import isfile, join

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
    #process every video in algonauts folder
    video_folder_path = 'AlgonautsVideos268_All_30fpsmax'
    all_files = [f for f in listdir(video_folder_path) if isfile(join(video_folder_path, f))]


    no_audio_vids = []
    
    for count, video in enumerate(all_files):
        try:
            # get a video path
            video_path = os.path.join(video_folder_path, video)
            print(video_path)
            # process the video for RAFT, MFCC, and BDCN
            print("Starting RAFT preprocessing...")
            process_video(video_path, 'RAFT')
            print("Starting BDCN preprocessing...")
            process_video(video_path, 'BDCN')
            print("Starting MFCC preprocessing...")
            process_video(video_path, 'MFCC')
            # log percentage progress to console
            print('{}/{}'.format(count, len(all_files)))
            # estimate how much time is left
            time_left = (len(all_files) - count) * (1 / (count + 1))
            print('Estimated time left: {} seconds'.format(time_left))
        except:
            # if the video has no audio, add it to a list
            print("Error processing video: {}".format(video))
            no_audio_vids.append(video_path)
            print("Starting BDCN preprocessing...")
            process_video(video_path, 'BDCN')
            continue
    
    # print out the number of videos that had no audio
    print("Number of videos with no audio: {}".format(len(no_audio_vids)))
    # print out percentage of videos that had no audio
    print("Percentage of videos with no audio: {}".format(len(no_audio_vids) / len(all_files)))

    #write the list of videos with no audio to a file
    with open('no_audio_vids.txt', 'w') as f:
        for item in no_audio_vids:
            f.write("%s\n" % item)
    

    # # remove all folders of videos that don't have audio
    # for video in no_audio_vids:
    #     os.remove(video)


if __name__ == "__main__":
    main()
