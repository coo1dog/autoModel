import logging
import sys
import os

def setup_logging(console_level=logging.INFO, log_file="automl.log", file_level=logging.DEBUG):
    """
    Set up the root logger for the application.

    This function configures the root logger to send messages to both the console
    and a log file.

    Args:
        console_level (int): The logging level for the console output.
        log_file (str): The name of the file to log to.
        file_level (int): The logging level for the file output.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

    # Prevent adding handlers multiple times if this function is called again
    if len(logger.handlers) > 0:
        return

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a handler for console output
    # Force stdout to use utf-8 encoding to fix display issues in Windows subprocesses
    if sys.platform.startswith('win'):
        import io
        if isinstance(sys.stdout, io.TextIOWrapper):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except Exception:
                pass

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a handler for file output
    # Check environment variable to prevent double logging (if stdout is already redirected to file)
    if log_file and not os.environ.get("NO_LOG_FILE"):
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8') # 'w' to overwrite the log on each run
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

