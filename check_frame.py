import cv2
import typing
from ml_serving.drivers import driver
from skimage.metrics import structural_similarity

import detect


class CheckFrameException(Exception):
    pass


_face_detect_driver: typing.Optional[driver.ServingDriver] = None
_face_detect_threshold: typing.Optional[float] = None
_change_scene_threshold: typing.Optional[float] = None

default_face_detect_model = '/opt/intel/openvino/deployment_tools/intel_models/'\
                            'face-detection-adas-0001/FP32/face-detection-adas-0001.xml'


def initialize(face_detect_model_path=None, face_detect_threshold=.5, change_scene_threshold=.5):
    global _face_detect_driver, _change_scene_threshold, _face_detect_threshold
    if face_detect_model_path is None:
        face_detect_model_path = default_face_detect_model
    _face_detect_driver = detect.get_driver(face_detect_model_path, "face detection")
    _change_scene_threshold = change_scene_threshold
    _face_detect_threshold = face_detect_threshold


def is_correct(frame, previous=None):
    if _change_scene_threshold is not None and previous is not None:
        i1 = cv2.resize(frame, (100, 100), interpolation=cv2.INTER_AREA)
        i2 = cv2.resize(previous, (100, 100), interpolation=cv2.INTER_AREA)
        score, _ = structural_similarity(i1, i2, full=True, multichannel=True)
        if 1 - score > _change_scene_threshold:
            raise CheckFrameException("large difference with previous frame, probably another scene")

    if _face_detect_driver is not None:
        faces = detect.detect_bboxes(_face_detect_driver, frame, threshold=_face_detect_threshold)
        if len(faces) != 1:
            raise CheckFrameException("detected %d faces, expected 1" % len(faces))
