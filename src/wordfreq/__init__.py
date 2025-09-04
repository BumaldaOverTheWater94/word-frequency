from loguru import logger
import sys

# remove default sink
logger.remove()
logger.add(
    sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>"
)


