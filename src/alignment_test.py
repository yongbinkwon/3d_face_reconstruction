import face_alignment
import cv2

alignment_model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
img_orig = cv2.imread("JianzhuGuo.jpg")
preds = alignment_model.get_landmarks(img_orig[:, :, ::-1])
print(preds[0].dtype)
