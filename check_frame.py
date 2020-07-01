import typing
from skimage.metrics import structural_similarity

import detect
from util import image_resize


class CheckFrameException(Exception):
    pass


_face_detect_threshold: typing.Optional[float] = None
_change_scene_threshold: typing.Optional[float] = None


def initialize(models_dir="./models", face_detect_threshold=.5, change_scene_threshold=.5):
    global _face_detect_driver, _change_scene_threshold, _face_detect_threshold
    detect.init_detector(models_dir)
    _change_scene_threshold = change_scene_threshold
    _face_detect_threshold = face_detect_threshold


def is_correct(frame, previous=None):
    if _change_scene_threshold is not None and previous is not None:
        score, _ = structural_similarity(_small_img(frame), _small_img(previous), full=True, multichannel=True)
        if 1 - score > _change_scene_threshold:
            raise CheckFrameException("large difference with previous frame, probably another scene")

    faces = detect.detect_faces(frame, threshold=_face_detect_threshold)
    if len(faces) != 1:
        raise CheckFrameException("detected %d faces, expected 1" % len(faces))


def _small_img(image, width=100, height=100):
    resized, _ = image_resize(image, width=width, height=height, contain_proportions=False)
    return resized
