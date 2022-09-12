import sys
from functools import partialmethod
from loguru import logger

STDOUT_LEVELS = ["GENERATION", "PROMPT"]
INIT_LEVELS = ["INIT", "INIT_OK", "INIT_WARN", "INIT_ERR"]
MESSAGE_LEVELS = ["MESSAGE"]

def is_stdout_log(record):
    if record["level"].name in STDOUT_LEVELS:
        return(True)
    return(False)

def is_init_log(record):
    if record["level"].name in INIT_LEVELS:
        return(True)
    return(False)

def is_msg_log(record):
    if record["level"].name in MESSAGE_LEVELS:
        return(True)
    return(False)

def is_stderr_log(record):
    if record["level"].name not in STDOUT_LEVELS + INIT_LEVELS + MESSAGE_LEVELS:
        return(True)
    return(False)

logfmt = "<level>{level: <10}</level> | <green>{name}</green>:<green>{function}</green>:<green>{line}</green> - <level>{message}</level>"
genfmt = "<level>{level: <10}</level> @ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"
initfmt = "<magenta>INIT      </magenta> | <level>{extra[status]: <10}</level> | <magenta>{message}</magenta>"
msgfmt = "<level>{level: <10}</level> | <level>{message}</level>"

logger.level("GENERATION", no=24, color="<cyan>")
logger.level("PROMPT", no=23, color="<yellow>")
logger.level("INIT", no=31, color="<white>")
logger.level("INIT_OK", no=31, color="<green>")
logger.level("INIT_WARN", no=31, color="<yellow>")
logger.level("INIT_ERR", no=31, color="<red>")
logger.level("MESSAGE", no=51, color="<green>")

logger.__class__.generation = partialmethod(logger.__class__.log, "GENERATION")
logger.__class__.prompt = partialmethod(logger.__class__.log, "PROMPT")
logger.__class__.init = partialmethod(logger.__class__.log, "INIT")
logger.__class__.init_ok = partialmethod(logger.__class__.log, "INIT_OK")
logger.__class__.init_warn = partialmethod(logger.__class__.log, "INIT_WARN")
logger.__class__.init_err = partialmethod(logger.__class__.log, "INIT_ERR")
logger.__class__.message = partialmethod(logger.__class__.log, "MESSAGE")

config = {
    "handlers": [
        {"sink": sys.stderr, "format": logfmt, "colorize":True, "filter": is_stderr_log},
        {"sink": sys.stdout, "format": genfmt, "level": "PROMPT", "colorize":True, "filter": is_stdout_log},
        {"sink": sys.stdout, "format": initfmt, "level": "INIT", "colorize":True, "filter": is_init_log},
        {"sink": sys.stdout, "format": msgfmt, "level": "MESSAGE", "colorize":True, "filter": is_msg_log}
    ],
}
logger.configure(**config)
