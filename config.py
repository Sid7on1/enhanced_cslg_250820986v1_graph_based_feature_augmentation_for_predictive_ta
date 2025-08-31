# config.py
"""
Model configuration file for the computer_vision project.

This file contains the necessary configuration settings for the project,
including model parameters, data loading, and logging.
"""

import logging
import os
from typing import Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("config.log"),
        logging.StreamHandler()
    ]
)

class Config:
    """
    Model configuration class.

    This class contains the necessary configuration settings for the project,
    including model parameters, data loading, and logging.
    """

    def __init__(self):
        """
        Initialize the configuration object.

        This method sets up the default configuration settings for the project.
        """
        self.model_params = self._get_model_params()
        self.data_loading = self._get_data_loading_params()
        self.logging = self._get_logging_params()

    def _get_model_params(self) -> Dict:
        """
        Get the model parameters.

        Returns:
            Dict: Model parameters.
        """
        model_params = {
            "model_name": "LG_2508.20986v1",
            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "momentum": 0.9
        }
        return model_params

    def _get_data_loading_params(self) -> Dict:
        """
        Get the data loading parameters.

        Returns:
            Dict: Data loading parameters.
        """
        data_loading_params = {
            "data_path": "/path/to/data",
            "data_format": "csv",
            "num_workers": 4
        }
        return data_loading_params

    def _get_logging_params(self) -> Dict:
        """
        Get the logging parameters.

        Returns:
            Dict: Logging parameters.
        """
        logging_params = {
            "log_level": "INFO",
            "log_file": "config.log",
            "console_log": True
        }
        return logging_params

    def get_config(self) -> Dict:
        """
        Get the configuration settings.

        Returns:
            Dict: Configuration settings.
        """
        config = {
            "model_params": self.model_params,
            "data_loading": self.data_loading,
            "logging": self.logging
        }
        return config

class ConfigValidator:
    """
    Configuration validator class.

    This class contains methods for validating the configuration settings.
    """

    def __init__(self, config: Config):
        """
        Initialize the configuration validator object.

        Args:
            config (Config): Configuration object.
        """
        self.config = config

    def validate_model_params(self) -> bool:
        """
        Validate the model parameters.

        Returns:
            bool: Validation result.
        """
        # Check if model_name is a string
        if not isinstance(self.config.model_params["model_name"], str):
            logging.error("Model name must be a string.")
            return False

        # Check if num_epochs is a positive integer
        if not isinstance(self.config.model_params["num_epochs"], int) or self.config.model_params["num_epochs"] <= 0:
            logging.error("Number of epochs must be a positive integer.")
            return False

        # Check if batch_size is a positive integer
        if not isinstance(self.config.model_params["batch_size"], int) or self.config.model_params["batch_size"] <= 0:
            logging.error("Batch size must be a positive integer.")
            return False

        # Check if learning_rate is a float
        if not isinstance(self.config.model_params["learning_rate"], float):
            logging.error("Learning rate must be a float.")
            return False

        # Check if weight_decay is a float
        if not isinstance(self.config.model_params["weight_decay"], float):
            logging.error("Weight decay must be a float.")
            return False

        # Check if momentum is a float
        if not isinstance(self.config.model_params["momentum"], float):
            logging.error("Momentum must be a float.")
            return False

        return True

    def validate_data_loading_params(self) -> bool:
        """
        Validate the data loading parameters.

        Returns:
            bool: Validation result.
        """
        # Check if data_path is a string
        if not isinstance(self.config.data_loading["data_path"], str):
            logging.error("Data path must be a string.")
            return False

        # Check if data_format is a string
        if not isinstance(self.config.data_loading["data_format"], str):
            logging.error("Data format must be a string.")
            return False

        # Check if num_workers is a positive integer
        if not isinstance(self.config.data_loading["num_workers"], int) or self.config.data_loading["num_workers"] <= 0:
            logging.error("Number of workers must be a positive integer.")
            return False

        return True

    def validate_logging_params(self) -> bool:
        """
        Validate the logging parameters.

        Returns:
            bool: Validation result.
        """
        # Check if log_level is a string
        if not isinstance(self.config.logging["log_level"], str):
            logging.error("Log level must be a string.")
            return False

        # Check if log_file is a string
        if not isinstance(self.config.logging["log_file"], str):
            logging.error("Log file must be a string.")
            return False

        # Check if console_log is a boolean
        if not isinstance(self.config.logging["console_log"], bool):
            logging.error("Console log must be a boolean.")
            return False

        return True

    def validate_config(self) -> bool:
        """
        Validate the configuration settings.

        Returns:
            bool: Validation result.
        """
        # Validate model parameters
        if not self.validate_model_params():
            return False

        # Validate data loading parameters
        if not self.validate_data_loading_params():
            return False

        # Validate logging parameters
        if not self.validate_logging_params():
            return False

        return True

def load_config() -> Config:
    """
    Load the configuration settings.

    Returns:
        Config: Configuration object.
    """
    config = Config()
    return config

def validate_config(config: Config) -> bool:
    """
    Validate the configuration settings.

    Args:
        config (Config): Configuration object.

    Returns:
        bool: Validation result.
    """
    validator = ConfigValidator(config)
    return validator.validate_config()

if __name__ == "__main__":
    config = load_config()
    logging.info("Configuration settings:")
    logging.info(config.get_config())
    if validate_config(config):
        logging.info("Configuration settings are valid.")
    else:
        logging.error("Configuration settings are invalid.")