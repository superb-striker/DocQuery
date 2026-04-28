import logging
from colorama import Fore, Style, init

# Initialize colorama : Automatically resets color after each print/log - without this, colors would “bleed” into future text.
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }
    # This method runs every time a log message is emitted :
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"   # Wraps message with color + reset : ensures only this message is colored

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = ColoredFormatter("[%(levelname)s] [%(name)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger