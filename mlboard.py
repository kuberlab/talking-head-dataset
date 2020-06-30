import logging

try:
    from mlboardclient.api import client
except ImportError:
    client = None

mlboard = None
mlboard_tried = False


def get():
    if client is None:
        return None
    global mlboard, mlboard_tried
    if not mlboard_tried:
        mlboard_tried = True
        mlboard = client.Client()
        try:
            mlboard.apps.get()
        except Exception:
            mlboard = None
            logging.info('Do not use mlboard.')
        else:
            logging.info('Use mlboard parameters logging.')
    return mlboard
