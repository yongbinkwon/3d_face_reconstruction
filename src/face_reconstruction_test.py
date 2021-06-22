from face_reconstruction import Face_reconstructor

if __name__ == '__main__':
    reconstuctor = Face_reconstructor()
    reconstuctor.reconstruct("./test_images/00003/front.jpg", "./test_images/00003/front.obj")