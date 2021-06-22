import numpy as np
import xml.etree.ElementTree as et
from skimage.io import imread, imsave
import skimage.transform
import face3d
from face3d import mesh
from skimage import io

def strip_obj_string(lines, depth, ind_start, ind_end):
    '''
    iterates over read obj lines and produces a numpy array with vertices in [x,y,z] format
    '''
    array = np.zeros((len(lines), depth+1))
    for i, line in enumerate(lines):
        sub_array = np.append(np.array(line[ind_start:ind_end].split(' ')), 1.0)
        array[i] = sub_array
    return array

def apply_similarity_and_projection_transform(vertices, R, t3d, P, s=1.):
    '''
    vertices = 3D vertices you want to transform
    R = rotation matrix for vertices
    t3d = translation vertex for x,y,z
    P = perspective matrix
    s = scale
    '''
    # similarity transform
    t_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    # projection
    t_vertices_h = np.hstack((t_vertices, np.ones((t_vertices.shape[0], 1))))
    projected_vertices = t_vertices_h.dot(P.T)

    return projected_vertices

def from_projected_vertices_to_image_vertices(projected_vertices, height=256, width=256, depth=256):
    '''
    Defines and applies a viewport transform
    divide the projected vertices on their w
    we normalize z to begin at 0
    '''
    viewport_matrix = np.array([
        [(width - 0) / 2, 0, 0, (0 + width) / 2],
        [0, (height - 0) / 2, 0, (0 + height) / 2],
        [0, 0, (depth - 0) / 2, (0 + depth) / 2],
        [0, 0, 0, 1]
    ])
    image_vertices_h = projected_vertices.dot(viewport_matrix.T)
    image_vertices = np.divide(image_vertices_h[:, :3], image_vertices_h[:, 3, np.newaxis])  # divide by w

    image_vertices[:, 2] = np.max(image_vertices[:, 2]) - image_vertices[:, 2]  # substract z by max z value

    return image_vertices


def get_image_vertices_from_facegen(obj_vertices, img_path, save_image=False):
    xml_settings_cam_path = img_path.replace('.png', '_cam.xml')
    cam_settings_file = et.parse(xml_settings_cam_path)
    cam_settings = cam_settings_file.getroot()
    xml_val_cam = cam_settings.find('val')
    xml_frustum = xml_val_cam.find('frustum')
    hw = float(xml_frustum.find('nearHalfWidth').text)
    hh = float(xml_frustum.find('nearHalfHeight').text)
    n0 = float(xml_frustum.find('nearDist').text)
    f0 = float(xml_frustum.find('farDist').text)
    r0 = hw
    l0 = -hw
    t0 = -hh
    b0 = hh
    proj_matrix = np.array([
        [2*n0/(r0-l0), 0, (r0+l0)/(r0-l0), 0],
        [0, 2*n0/(t0-b0), (t0+b0)/(t0-b0), 0],
        [0, 0, -(f0+n0)/(f0-n0), -(2*f0*n0)/(f0-n0)],
        [0, 0, -1, 0]
    ])
    
    with open(f"{img_path.replace('.png', '_mvm.txt')}", "r") as f:
        mvm = np.array([float(entry) for entry in f.readline().strip().split(" ")]).reshape(4, 4)

    transformed_vertices = obj_vertices.dot(mvm.T)
    projected_vertices = transformed_vertices.dot(proj_matrix.T)
    image_vertices = from_projected_vertices_to_image_vertices(projected_vertices)

    # save output image
    if (save_image):
        img_out_path = img_path.replace('.png', '_projected.png')
        img_vertices_clipped = np.clip(image_vertices, 0, 256)  # clip vertice coords to fit image bounds
        img = imread(img_path) / 255.
        img_reshape = skimage.transform.resize(img, (256, 256))
        image_out = img_reshape.copy().astype(np.float32)

        for i, vertex in enumerate(img_vertices_clipped):
            ind = np.round(vertex).astype(np.int8)
            image_out[ind[1]][ind[0]] = [1, 0, ind[2] / 256, 1]
        imsave(img_out_path, image_out)

    return image_vertices

def generate_posmap_facegen_bfm(bfm, uv_coords, img_path, obj_path, pos_map_path, width=256, height=256, save_image=False):
    with open(obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -1)
    image_vertices = get_image_vertices_from_facegen(vertices_array, img_path)
    image_vertices[:, 1] = height - image_vertices[:, 1] - 1

    # crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0,
                    bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top) / 2
    size = int(old_size * 1.5)

    # randomize the cropping size
    marg = old_size * 0.1
    t_x = np.random.rand() * marg * 2 - marg
    t_y = np.random.rand() * marg * 2 - marg
    center[0] = center[0] + t_x
    center[1] = center[1] + t_y
    size = size * (np.random.rand() * 0.2 + 0.9)

    # crop and record the transform parameters
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2,
                                                                    center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, height - 1], [width - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)

    # transform face position(image vertices) along with 2d facial image
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z

    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, height, width, c=3)
    uv_position_map = uv_position_map.astype(np.float16)

    if save_image:
        io.imsave(pos_map_path.replace('.npy', '.png'), uv_position_map / np.amax(uv_position_map))

    # save position map
    #print(pos_map_path, uv_position_map.shape)
    np.save(pos_map_path, uv_position_map)

    # return cropping transform
    return tform

if __name__=='__main__':
    with open("/mnt/lhome/lhome/yongbk/facegen/subjects/2/subject_2.obj") as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -1)
    image_vertices = get_image_vertices_from_facegen(vertices_array, "/mnt/lhome/lhome/yongbk/facegen/subjects/2/render_0.png")
