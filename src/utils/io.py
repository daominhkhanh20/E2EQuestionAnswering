import json
import logging
import yaml

logger = logging.getLogger(__name__)


def load_json_data(path_file: str):
    with open(path_file, 'r') as file:
        data = json.load(file)
    return data


def write_json_file(data: dict, path_file: str):
    with open(path_file, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    logger.info(f"Save data in {path_file}")


def load_yaml_file(path_file: str):
    with open(path_file) as file:
        data = yaml.load(file, Loader=yaml.loader.SafeLoader)
    return data