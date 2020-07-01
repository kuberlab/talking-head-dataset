import logging

import torch

import face_alignment
from util import image_resize


fa = None


def init_detector(
        models_dir,
        face_detector='sfd',
):
    global fa
    if fa:
        raise RuntimeError("unable to reinitialize face_detector")
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info("torch device: CUDA, "
                    f"device name: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logging.info(f"torch device: CPU")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D,
        face_detector=face_detector,
        device=device,
        models_dir=models_dir,
    )


def _check_fa():
    if not fa:
        raise RuntimeError("face alignment not initialized")


def detect_faces(image, threshold=.5):

    _check_fa()

    image_resized, aspects = image_resize(
        image,
        width=300, height=300,
        contain_proportions=False,
    )

    with torch.no_grad():
        faces = fa.face_detector.detect_from_image(image_resized[..., ::-1].copy(), threshold=threshold)

    return faces
