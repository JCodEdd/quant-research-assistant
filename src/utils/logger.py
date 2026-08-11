from loguru import logger

logger.remove()  # Remove the default logger

logger.add(
    "logs/app.log",  # Log file path
    rotation="10 MB",  # Rotate log files when they reach 10 MB
    level="INFO",  # Log level
)

logger.add(
    lambda msg: print(msg, end=""),  # Print logs to console
    level="INFO",  # Log level
)