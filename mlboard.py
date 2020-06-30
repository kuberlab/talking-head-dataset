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


def update_task_info(data, app_name=None,
                         task_name=None, build_id=None, fail_on_error=False):
    m = get()
    if m is not None:
        m.update_task_info(data, app_name, task_name, build_id, fail_on_error)