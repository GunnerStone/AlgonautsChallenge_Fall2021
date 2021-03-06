import os
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import cv2
import numpy as np
import torch

from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz

def frame_preprocess(frame, device):
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)
    frame = frame.to(device)
    return frame


def vizualize_flow(img, flo, save, counter, args):
    # permute the channels and change device is necessary
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    flo = cv2.cvtColor(flo, cv2.COLOR_RGB2BGR)

    # concatenate, save and show images
    #img_flo = np.concatenate([img, flo], axis=0)

    img_flo = flo
    
    #remove extension from video name
    video_name = args.video.split('.')[0]
    # get the base video name
    base_name = os.path.basename(video_name)

    if save:
        # print("Saving RAW & RAFT frames")
        # print("video name: "+str(video_name))
        cv2.imwrite("AlgonautsVideos268_Preprocessed/"+str(base_name)+f"/RAW/frame_{str(counter)}.png", img)
        cv2.imwrite("AlgonautsVideos268_Preprocessed/"+str(base_name)+f"/RAFT/frame_{str(counter)}.png", img_flo)
        #print out full directory of where we just saved the image
        # print(f"AlgonautsVideos268_Preprocessed/{base_name}/RAW/frame_{str(counter)}.png")
        # print(f"AlgonautsVideos268_Preprocessed/{base_name}/RAFT/frame_{str(counter)}.png")
    # cv2.imshow("Optical Flow", img_flo / 255.0)
    # k = cv2.waitKey(25) & 0xFF
    # if k == 27:
    #     return False
    return True


def get_cpu_model(model):
    new_model = OrderedDict()
    # get all layer's names from model
    for name in model:
        # create new name and update new model
        new_name = name[7:]
        new_model[new_name] = model[name]
    return new_model


def inference(args):
    # get the RAFT model
    model = RAFT(args)
    # load pretrained weights
    pretrained_weights = torch.load(args.model)

    #get the base name of the video
    base_name = os.path.basename(args.video)
    #get the name of the video without the extension
    base_name = base_name.split('.')[0]
    # print (base_name)

    save = args.save
    if save:
        if not os.path.exists("AlgonautsVideos268_Preprocessed"):
            os.mkdir("AlgonautsVideos268_Preprocessed")
        if not os.path.exists("AlgonautsVideos268_Preprocessed/"+str(base_name)):
            os.mkdir("AlgonautsVideos268_Preprocessed/"+str(base_name))
        if not os.path.exists("AlgonautsVideos268_Preprocessed/"+str(base_name)+"/RAFT"):
            os.mkdir("AlgonautsVideos268_Preprocessed/"+str(base_name)+"/RAFT")
        if not os.path.exists("AlgonautsVideos268_Preprocessed/"+str(base_name)+"/RAW"):
            os.mkdir("AlgonautsVideos268_Preprocessed/"+str(base_name)+"/RAW")

    if torch.cuda.is_available():
        device = "cuda"
        # parallel between available GPUs
        model = torch.nn.DataParallel(model)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)
        model.to(device)
    else:
        device = "cpu"
        # change key names for CPU runtime
        pretrained_weights = get_cpu_model(pretrained_weights)
        # load the pretrained weights into model
        model.load_state_dict(pretrained_weights)

    # change model's mode to evaluation
    model.eval()

    video_path = args.video
    # print the path to video
    # print(video_path)
    # capture the video and get the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame_1 = cap.read()


    if frame_1.shape[2] % 8 != 0:
        frame_1 = cv2.resize(frame_1, (int(frame_1.shape[1] / 8) * 8, int(frame_1.shape[0] / 8) * 8))
    # print(frame_1.shape)
    # frame preprocessing
    frame_1 = frame_preprocess(frame_1, device)
    counter = 0
    with torch.no_grad():
        while True:
            # read the next frame
            ret, frame_2 = cap.read()
            if not ret:
                break

            if frame_2.shape[2] % 8 != 0:
                frame_2 = cv2.resize(frame_2, (int(frame_2.shape[1] / 8) * 8, int(frame_2.shape[0] / 8) * 8))

            # preprocessing
            frame_2 = frame_preprocess(frame_2, device)

            

            
            # predict the flow
            flow_low, flow_up = model(frame_1, frame_2, iters=args.iters, test_mode=True)

            # transpose the flow output and convert it into numpy array
            ret = vizualize_flow(frame_1, flow_up, save, counter, args)
            if not ret:
                break
            frame_1 = frame_2
            counter += 1


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint", default="RAFT/models/raft-things.pth")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--video", type=str, default="testvid.mp4")
    parser.add_argument("--save", action="store_true", help="save demo frames")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    #print out the --video argument
    print(parser.parse_args().video)


    args = parser.parse_args()
    inference(args)


if __name__ == "__main__":
    main()
