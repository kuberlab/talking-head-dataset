import logging
import os

import cv2
from ml_serving.drivers import driver

import numpy as np


def get_driver(model_path: str, description: str = None, driver_name: str = None, **kwargs) -> driver.ServingDriver:
    path_device = model_path.split(',')
    if len(path_device) > 1:
        kwargs['device'] = path_device[1]
        model_path = path_device[0]

    if not driver_name:
        if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, 'saved_model.pb')):
            driver_name = 'tensorflow'
        elif model_path.endswith('.pth'):
            driver_name = 'pytorch'
        else:
            driver_name = 'openvino'
    if description:
        logging.info("Load %s %s model from %s..." % (description, driver_name, model_path))
    else:
        logging.info("Load undescribed %s model from %s..." % (driver_name, model_path))
    drv = driver.load_driver(driver_name)
    d = drv()
    d.load_model(model_path, **kwargs)
    return d


def detect_bboxes(drv: driver.ServingDriver, rgb_frame: np.ndarray, threshold: float = 0.5, offset=(0, 0)):
    if drv is None:
        return None
    if drv.driver_name == 'tensorflow':
        return _detect_bboxes_tensorflow(drv, rgb_frame, threshold, offset, only_class=1)  # only_class - for persons
    elif drv.driver_name == 'openvino':
        bgr_frame = rgb_frame[:, :, ::-1]
        return _detect_bboxes_openvino(drv, bgr_frame, threshold, offset)
    else:
        raise ValueError('unknown driver name {}'.format(drv.driver_name))


def _detect_bboxes_openvino(drv: driver.ServingDriver, bgr_frame: np.ndarray,
                            threshold: float = 0.5, offset=(0, 0)):
    # Get boxes shaped [N, 5]:
    # xmin, ymin, xmax, ymax, confidence
    input_name, input_shape = list(drv.inputs.items())[0]
    output_name = list(drv.outputs)[0]
    inference_frame = cv2.resize(bgr_frame, tuple(input_shape[:-3:-1]), interpolation=cv2.INTER_AREA)
    inference_frame = np.transpose(inference_frame, [2, 0, 1]).reshape(input_shape)
    outputs = drv.predict({input_name: inference_frame})
    output = outputs[output_name]
    output = output.reshape(-1, 7)
    bboxes_raw = output[output[:, 2] > threshold]
    # Extract 5 values
    boxes = bboxes_raw[:, 3:7]
    confidence = np.expand_dims(bboxes_raw[:, 2], axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)
    # Assign confidence to 4th
    # boxes[:, 4] = bboxes_raw[:, 2]
    boxes[:, 0] = boxes[:, 0] * bgr_frame.shape[1] + offset[0]
    boxes[:, 2] = boxes[:, 2] * bgr_frame.shape[1] + offset[0]
    boxes[:, 1] = boxes[:, 1] * bgr_frame.shape[0] + offset[1]
    boxes[:, 3] = boxes[:, 3] * bgr_frame.shape[0] + offset[1]
    return boxes


def _detect_bboxes_tensorflow(drv: driver.ServingDriver, frame: np.ndarray,
                              threshold: float = 0.5, offset=(0, 0), only_class=None):
    input_name, input_shape = list(drv.inputs.items())[0]
    inference_frame = np.expand_dims(frame, axis=0)
    outputs = drv.predict({input_name: inference_frame})
    boxes = outputs["detection_boxes"].copy().reshape([-1, 4])
    scores = outputs["detection_scores"].copy().reshape([-1])
    scores = scores[np.where(scores > threshold)]
    boxes = boxes[:len(scores)]
    if only_class is not None:
        classes = np.int32((outputs["detection_classes"].copy())).reshape([-1])
        classes = classes[:len(scores)]
        boxes = boxes[classes == only_class]
        scores = scores[classes == only_class]
    boxes[:, 0] *= frame.shape[0] + offset[0]
    boxes[:, 2] *= frame.shape[0] + offset[0]
    boxes[:, 1] *= frame.shape[1] + offset[1]
    boxes[:, 3] *= frame.shape[1] + offset[1]
    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # .astype(int)

    # add probabilities
    confidence = np.expand_dims(scores, axis=0).transpose()
    boxes = np.concatenate((boxes, confidence), axis=1)

    return boxes
