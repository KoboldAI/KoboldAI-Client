import sys
from functools import partialmethod
from loguru import logger

STDOUT_LEVELS = ["GENERATION", "PROMPT"]

def is_stdout_log(record):
    if record["level"].name in STDOUT_LEVELS:
        return(True)
    return(False)

def is_stderr_log(record):
    if record["level"].name not in STDOUT_LEVELS:
        return(True)
    return(False)

logfmt = "<level>{level: <10}</level> | <green>{name}</green>:<green>{function}</green>:<green>{line}</green> - <level>{message}</level>"
genfmt = "<level>{level: <10}</level> @ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"
promptfmt = "<level>{level: <10}</level> @ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

new_level = logger.level("GENERATION", no=22, color="<cyan>")
new_level = logger.level("PROMPT", no=21, color="<yellow>")

logger.__class__.generation = partialmethod(logger.__class__.log, "GENERATION")
logger.__class__.prompt = partialmethod(logger.__class__.log, "PROMPT")

config = {
    "handlers": [
        {"sink": sys.stderr, "format": logfmt, "colorize":True, "filter": is_stderr_log},
        {"sink": sys.stdout, "format": genfmt, "level": "PROMPT", "colorize":True, "filter": is_stdout_log},
    ],
}
logger.configure(**config)
