from image_synthesis import Synthesize
from tqdm import tqdm
import glob

if __name__ == '__main__':
    synthesizer = Synthesize()
    images = glob.glob("image_synthesis/model_fitting/example/Images/*.jpg")
    for filename in tqdm(images):
        synthesizer.synthesize_image(filename)
    #synthesizer.synthesize_image("image_synthesis/model_fitting/example/Images/hei2.jpg")