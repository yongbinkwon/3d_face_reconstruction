import scipy.io
import numpy as np
from os import path

if __name__=='__main__':
    mat = scipy.io.loadmat(path.expanduser("Data/BFM/BFM.mat"))['model'][0, 0]
    BFM_shape = mat['shapeMU']
    BFM_triangles = np.vstack((mat['tri'].T, mat['tri_mouth'].T))

    uv_mat = np.around(scipy.io.loadmat(path.expanduser("Data/BFM/BFM_UV.mat"))['UV'], 8)
    count = 0
    print(len(BFM_shape))
    
    """
    with open("meanBFM.obj", "w") as f:
        for i in range (0, len(BFM_shape), 3):
            f.write(f"v {BFM_shape[i][0]} {BFM_shape[i+1][0]} {BFM_shape[i+2][0]}\n")
            count += 1
        for uv_coord in uv_mat:
            f.write(f"vt {uv_coord[0]} {uv_coord[1]}\n")
        for triangle in BFM_triangles[:-1]:
            f.write(f"f {triangle[2]}/{triangle[2]} {triangle[1]}/{triangle[1]} {triangle[0]}/{triangle[0]}\n")
        last_triangle = BFM_triangles[-1]
        f.write(f"f {last_triangle[2]}/{last_triangle[2]} {last_triangle[1]}/{last_triangle[1]} {last_triangle[0]}/{last_triangle[0]}")
    """
"""
if __name__=='__main__':
    mat = scipy.io.loadmat(path.expanduser("~/yongbk/Downloads/PublicMM1/01_MorphableModel.mat"))
    BFM_shape = mat['shapeMU']
    print(BFM_shape.shape)
    BFM_triangles = mat['tl']
    print(BFM_triangles.shape)

    uv_mat = scipy.io.loadmat(path.expanduser("~/yongbk/Downloads/BFM_UV.mat"))['UV']
    print(uv_mat.shape)

    with open("meanBFM.obj", "w") as f:
        for i in range (0, len(BFM_shape), 3):
            f.write(f"v {BFM_shape[i][0]} {BFM_shape[i+1][0]} {BFM_shape[i+2][0]}\n")
        for uv_coord in uv_mat:
            f.write(f"vt {uv_coord[0]} {uv_coord[1]}\n")
        for triangle in BFM_triangles[:-1]:
            f.write(f"f {triangle[2]}/{triangle[2]} {triangle[1]}/{triangle[1]} {triangle[0]}/{triangle[0]}\n")
        last_triangle = BFM_triangles[-1]
        f.write(f"f {last_triangle[2]}/{last_triangle[2]} {last_triangle[1]}/{last_triangle[1]} {last_triangle[0]}/{last_triangle[0]}")
"""