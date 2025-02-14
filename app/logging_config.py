
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, "logs")
agent_log_file_path = os.path.join(log_dir, "agent.log")

# Configure agent logger
agent_logger = logging.getLogger("agent_logger")
agent_logger.setLevel(logging.INFO)
agent_logger.addHandler(logging.FileHandler(agent_log_file_path, mode='a'))
agent_logger.addHandler(logging.StreamHandler())

memory_log_file_path = os.path.join(log_dir, "memory.log")
# Configure memory logger
memory_logger = logging.getLogger("memory")
memory_logger.setLevel(logging.INFO)
memory_logger.addHandler(logging.FileHandler(memory_log_file_path, mode='a'))
memory_logger.addHandler(logging.StreamHandler())

indexing_log_file_path = os.path.join(log_dir, "indexing.log")
# Configure indexing logger
indexing_logger = logging.getLogger("indexing")
indexing_logger.setLevel(logging.INFO)
indexing_logger.addHandler(logging.FileHandler(indexing_log_file_path, mode='a'))
indexing_logger.addHandler(logging.StreamHandler())

retriever_log_file_path = os.path.join(log_dir, "retriever.log")
# Configure retriever logger
retriever_logger = logging.getLogger("retriever")
retriever_logger.setLevel(logging.INFO)
retriever_logger.addHandler(logging.FileHandler(retriever_log_file_path, mode='a'))
retriever_logger.addHandler(logging.StreamHandler())

service_log_file_path = os.path.join(log_dir, "service.log")
# Configure service logger
service_logger = logging.getLogger("service")
service_logger.setLevel(logging.INFO)
service_logger.addHandler(logging.FileHandler(service_log_file_path, mode='a'))
service_logger.addHandler(logging.StreamHandler())