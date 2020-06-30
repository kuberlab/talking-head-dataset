import logging


def init_logging():
    logging.addLevelName(logging.INFO, 'INFO')
    logging.addLevelName(logging.WARNING, 'WARN')
    logging.addLevelName(logging.ERROR, 'ERRO')
    logging.addLevelName(logging.CRITICAL, 'CRIT')
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d: %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
        ],
    )
