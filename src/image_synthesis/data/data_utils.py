import torch

def get_input(image, M, render, yaw_pose, param):
    real_image = image
    rotated_meshs = []
    rotated_landmarks_list = []
    original_angles_list = []
    rotated_landmarks_list_106 = []
    real_images = []
    rotated_mesh, rotate_landmarks, original_angles, rotate_landmarks_106\
        = render.rotate_render(param, real_image, M, yaw_pose=yaw_pose)
    rotated_meshs.append(rotated_mesh)
    rotated_landmarks_list.append(rotate_landmarks)
    rotated_landmarks_list_106.append(rotate_landmarks_106)
    original_angles_list.append(original_angles)
    real_images.append(real_image)
    rotated_meshs = torch.cat(rotated_meshs, 0)
    rotated_landmarks_list = torch.cat(rotated_landmarks_list, 0)
    rotated_landmarks_list_106 = torch.cat(rotated_landmarks_list_106, 0)
    original_angles_list = torch.cat(original_angles_list, 0)
    output = {}
    real_image = real_image * 2 - 1
    rotated_meshs = rotated_meshs * 2 - 1
    output['image'] = real_image.cpu()
    output['rotated_mesh'] = rotated_meshs.cpu()
    output['rotated_landmarks'] = rotated_landmarks_list.cpu()
    output['rotated_landmarks_106'] = rotated_landmarks_list_106.cpu()
    output['original_angles'] = original_angles_list.cpu()
    return output
