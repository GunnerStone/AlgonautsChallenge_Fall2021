import os
import imageio
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

#store all png filenames to this list
filenames = natural_sort((fn for fn in os.listdir('C:/Users/gunner/Desktop/Algonauts/demo_frames') if fn.endswith('.png')))

#get every nth iteration
N = 1
filenames = filenames[0::N]
with imageio.get_writer('C:/Users/gunner/Desktop/Algonauts/Presentation_Media/movie_non_epilepsy.gif', mode='I', duration=.1) as writer:
    for filename in filenames:
        image = imageio.imread('C:/Users/gunner/Desktop/Algonauts/demo_frames/'+filename)
        writer.append_data(image)