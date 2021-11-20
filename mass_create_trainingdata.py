from algo_interface import get_label
from helper_functions import get_np_array_from_image
import os
import numpy as np

def get_video_names(containing_folder="./AlgonautsVideos268_All_30fpsmax"):
    """
    Returns a list of all video names
    """
    import os
    video_names = []
    for file in os.listdir(containing_folder):
        if file.endswith(".mp4"):
            video_names.append(file)
    #remove file extension
    for i in range(len(video_names)):
        video_names[i] = video_names[i][:-4]
    return video_names

def get_video_id(video_name):
    """
        Returns video id from first 4 characters of video name and returns an int
    """
    # video id is first 4 characters of video name minus 1 because they have to be 0 indexed
    return int(video_name[0:4]) - 1

def create_trainingdata():
    """
    Creates training data for the model
    """
    

    subjects = ["sub01","sub02","sub03","sub04","sub05","sub06","sub07","sub08","sub09","sub10"]
    # rois = ["V1", "V2","V3", "V4", "LOC", "EBA", "FFA","STS", "PPA"]
    rois = ["FFA","STS", "PPA"]
    techniques = ["BDCN","RAW","RAFT","MFCC"]

    # create folders in Gunners_training_data for each ROI if they don't exist
    for roi in rois:
        if not os.path.exists("Gunners_training_data/"+str(roi)):
            os.makedirs("Gunners_training_data/"+str(roi))
    
    # create subfolders in each ROI folder for each technique if they don't exist
    for technique in techniques:
        for roi in rois:
            if not os.path.exists("Gunners_training_data/"+str(roi)+"/"+technique):
                os.makedirs("Gunners_training_data/"+str(roi)+"/"+technique)

    training_data = []

    # create training data for each ROI and technique
    for roi in rois:
        for technique in techniques:

            video_names = get_video_names()

            #only take first 1000 videos
            for video_name in video_names:

                # get video id
                video_id = get_video_id(video_name)

                # if video id is 1000 or greater, skip the sample, as there is no label for it
                if video_id >= 1000:
                    continue

                 # get concatenated numpy array of image technique
                folder_path = "./AlgonautsVideos268_Preprocessed/"+video_name
                
                image = get_np_array_from_image(folder_path=folder_path,type=technique,img_size=32)

                # if image variable is None, that means MFCC doesnt exist (no audio) so skip this sample
                if image is None:
                    continue

                # loop through subjects last so we dont muddy training / validation data when truncating later
                for subject in subjects:      
                    # get subjects voxelwise response for sample (label)
                    label = get_label(video_name, subject, roi)
                   
                    # append to training data
                    training_data.append([image, label])
                    # save training data
                    np.save("Gunners_training_data/"+str(roi)+"/"+technique+"/"+"{}_{}_{}_{}".format(video_id,subject,roi,technique),np.array(training_data,dtype=object))
                    # clear training data
                    training_data = []
            print("Done processing with: " + str(roi) + " " +str(technique))
            
    return
def main():
    my_training_data = create_trainingdata()
    print(my_training_data.shape)
    print(my_training_data[0][0].shape) # image
    print(my_training_data[0][1].shape) # label


#run main function
if __name__ == "__main__":
    main()