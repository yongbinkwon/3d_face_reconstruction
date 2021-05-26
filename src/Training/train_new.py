from Networks import mobilenet_v2
from Networks.mobilenet_v2 import MobileNetv2_PRN

def main(args):
    model = MobileNetv2_PRN((256, 256, 6), args.alpha_mobilenet)
    return None


if __name__ == '__main__':
