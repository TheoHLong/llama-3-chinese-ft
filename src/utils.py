import logging
import os
import yaml
import torch
from typing import Dict, Any, Optional

def setup_logging(log_path: Optional[str] = None) -> None:
    """Setup logging configuration."""
    logging_config = {
        'version': 1,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'level': 'INFO'
            }
        },
        'root': {
            'handlers': ['console'],
            'level': 'INFO',
        }
    }
    
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging_config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'filename': log_path,
            'formatter': 'standard',
            'level': 'INFO'
        }
        logging_config['root']['handlers'].append('file')
    
    logging.config.dictConfig(logging_config)

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    memory_stats = {}
    for i in range(torch.cuda.device_count()):
        gpu_properties = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)    # Convert to GB
        
        memory_stats[f"gpu_{i}"] = {
            "name": gpu_properties.name,
            "total_memory": gpu_properties.total_memory / (1024**3),  # Convert to GB
            "allocated_memory": memory_allocated,
            "reserved_memory": memory_reserved
        }
    
    return memory_stats

def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create a directory for the experiment with incremental naming."""
    i = 1
    while True:
        dir_name = f"{experiment_name}_{i}"
        full_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        i += 1

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            logging.error(f"Error loading config file: {e}")
            raise

def save_yaml_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w', encoding='utf-8') as f:
        try:
            yaml.dump(config, f, default_flow_style=False)
        except yaml.YAMLError as e:
            logging.error(f"Error saving config file: {e}")
            raise

def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """Calculate model size and parameter count."""
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "parameter_count": param_count,
        "parameter_size_mb": param_size / 1024**2,
        "buffer_size_mb": buffer_size / 1024**2,
        "total_size_mb": size_all_mb
    }