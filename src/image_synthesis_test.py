from image_synthesis import Synthesize
import glob
from tqdm import tqdm

if __name__ == '__main__':
    synthesizer = Synthesize()
    
    images = glob.glob("image_synthesis/images/*.jpg")
    for i in range(4):
        for filename in tqdm(images):
            synthesizer.synthesize_image(filename)
    
    #synthesizer.synthesize_image("/mnt/lhome/lhome/yongbk/3DDFA_V2/examples/inputs/JianzhuGuo.jpg")

