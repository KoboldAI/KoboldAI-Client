import sys
from functools import partialmethod
from loguru import logger

def is_stdout_log(record):
    if record["level"].name == "GENERATION":
        return(True)
    return(False)
def is_stderr_log(record):
    if record["level"].name != "GENERATION":
        return(True)
    return(False)
logfmt = "<level>{level: <10}</level> | <green>{name}</green>:<green>{function}</green>:<green>{line}</green> - <level>{message}</level>"
genfmt = "<level>{level: <10}</level> @ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"
new_level = logger.level("GENERATION", no=21, color="<cyan>")
logger.__class__.generation = partialmethod(logger.__class__.log, "GENERATION")
config = {
    "handlers": [
        {"sink": sys.stderr, "format": logfmt, "colorize":True, "filter": is_stderr_log},
        {"sink": sys.stdout, "format": genfmt, "level": "GENERATION", "colorize":True, "filter": is_stdout_log},
    ],
}
logger.configure(**config)
