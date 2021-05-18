import scipy.io
from os import path

if __name__=='__main__':
    mat = scipy.io.loadmat(path.expanduser("~/yongbk/Downloads/PublicMM1/01_MorphableModel.mat"))
    meanBFM = mat['shapeMU']
    meanBFM_triangles = mat['tl']
    count = 0

    with open("meanBFM.obj", "w") as f:
        for i in range (0, len(meanBFM)-2, 3):
            f.write(f"v {meanBFM[i][0]} {meanBFM[i+1][0]} {meanBFM[i+2][0]}\n")
            count += 1
        for triangle in meanBFM_triangles:
            f.write(f"f {triangle[2]} {triangle[1]} {triangle[0]}\n")
    print(count)
