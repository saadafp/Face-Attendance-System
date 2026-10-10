import numpy as np
from insightface.utils.face_align import norm_crop

def align_face(img, landmarks, image_size=112):
    """
    norm_crop(img, landmark) → aligned
    aligned.astype(np.float32) → float32
    np.transpose(HWC→CHW)
    np.expand_dims(batch) → (1,3,112,112)
    :param img:
    :param landmarks:
    :param image_size:
    :return: blob
    """
    # Alignment
    aligned = norm_crop(img, landmarks, image_size=image_size)

    # Prepare for ONNX model
    blob = aligned.astype(np.float32)  # dtype
    blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
    blob = np.expand_dims(blob, axis=0)  # add batch dim: 1x3x112x112

    return blob




