import numpy as np
from glob import glob
import cv2

# create a helper function that takes all png files in a folder and concatenates them into a single numpy array
def load_images(folder_path, img_size=32, num_channels=3, dtype=np.float32, normalize=True):
    """Loads images from a folder.

    Args:
        folder_path: Path to a folder with png images.
        img_size: Size of the image.
        num_channels: Number of channels in the output image.
        dtype: Data type of the image array.
        normalize: Whether to normalize the images.

    Returns:
        An array with all the images.
    """
    images = []
    for filename in glob(folder_path + '/*.png'):
        if num_channels == 1:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(filename)
        if img is None:
            raise Exception("Could not read image: " + filename)       
        img = cv2.resize(img, (img_size, img_size))
        if num_channels == 1:
            #expand dims to (img_size, img_size, num_channels)
            img = np.expand_dims(img, axis=2)
        img = img.astype(dtype)
        if normalize:
            img = img / 255.0
        images.append(img)
    images = np.concatenate(images, axis=2)
    return images

def get_np_array_from_image(folder_path, type="RAFT", img_size=32, dtype=np.float32, normalize=True):
    # append type to folder path
    folder_path = folder_path + "/" + type
    if type == "RAFT":
        num_channels = 3
        images = load_images(folder_path,img_size=img_size,num_channels=num_channels)
        #only get the first 75 frames
        images = images[:,:,0:75*num_channels-3]
    elif type == "BDCN":
        num_channels = 1
        images = load_images(folder_path,img_size=img_size,num_channels=num_channels)
        

        #only get the first 75 frames
        images = images[:,:,0:75*num_channels]
    elif type == "RAW":
        num_channels = 3
        images = load_images(folder_path,img_size=img_size,num_channels=num_channels)
        #only get the first 75 frames
        images = images[:,:,0:75*num_channels]
    elif type == "MFCC":
        # load a .npy file from folder_path
        try:
            images = np.load(folder_path+"/MFCC.npy", allow_pickle=False)
        except:
            #print("Could not load {}/MFCC.npy".format(folder_path))
            return None
    return images    
    


def main():
    folder_path = "./AlgonautsVideos268_Preprocessed/0007_0-2-3-14056753023"
    images = get_np_array_from_image(folder_path,type="BDCN",img_size=32)
    print("Showing image")
    #show the first image
    cv2.imshow("image", images[0])
    cv2.waitKey(0)

    print(images.shape)

#run main function
if __name__ == "__main__":
    main()
