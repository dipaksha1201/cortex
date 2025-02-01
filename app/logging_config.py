
import logging
import os

# Get absolute path to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, "logs")
resoning_log_file_path = os.path.join(log_dir, "reasoning.log")

# Create logs directory
os.makedirs(log_dir, exist_ok=True)

# Configure reasoning logger
reasoning_logger = logging.getLogger("reasoning_logger")
reasoning_logger.setLevel(logging.INFO)
reasoning_logger.addHandler(logging.FileHandler(resoning_log_file_path, mode='a'))
reasoning_logger.addHandler(logging.StreamHandler())