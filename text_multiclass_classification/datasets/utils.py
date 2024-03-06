import os
import re

import requests

from text_multiclass_classification import logger


def download_data(source: str, destination: str) -> str:
    """Downloads a dataset, i.e. a csv file from a source
    url to a local destination path.

    Args:
        source (str): A url to the data file.
        destination (str): A local filepath to save
            the downloaded data.

    Returns:
        str: The path to the downloaded data.
    """
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination), exist_ok=True)

    response = requests.get(url=source)

    if response.status_code == 200:
        logger.info(f"Downloading data from {source} to {destination}")
        with open(file=destination, mode="wb") as f:
            f.write("category,title,description\n".encode("utf-8"))
            f.write(response.content)
    else:
        logger.error(f"Failed to download data. Status code: {response.status_code}")

    return destination


def basic_text_normalization(text: str) -> str:
    """Basic normalization for a provided text.

    Normalization includes
    - lowercasing
    - adding whitespace around punctuation symbols
    - replace multiple spaces with single space
    """
    text = text.lower()
    text = re.sub(pattern=r"([.,!?])", repl=r" \1 ", string=text)
    text = re.sub(pattern=r"[^a-zA-Z.,!?]+", repl=r" ", string=text)
    text = re.sub(pattern=r"\s+", repl=" ", string=text)
    return text
