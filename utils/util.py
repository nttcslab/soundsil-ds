import os
import yaml


def load_config_yaml(config_file_path):
    """load configuration yaml file from path

    Args:
        config_file_path (_type_): path to configuration file

    Returns:
        _type_: dictionary of configuration yaml contents
    """
    assert os.path.isfile(config_file_path)
    with open(config_file_path) as f:
        yaml_contents = yaml.safe_load(f)
    return yaml_contents
