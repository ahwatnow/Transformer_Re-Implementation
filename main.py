from datasets import load_dataset
from utils.py import preprocess 

import argparse
import logging
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="transformer")
parser.add_argument("--word_count", type=int, default=20000)
parser.add_argument("--source", type=str, default = "en")
parser.add_argument("--target", type=str, default="de")
parser.add_argument("--config_dir", type=str, default="config.yaml"), 
parser.add_argument("--num_gpus", type=int, default=2)

args = parser.parse_args()

# Load the IWSLT 2017 dataset for a specific language pair
dataset = load_dataset("iwslt2017", "iwslt2017-en-de")

# Access specific splits of the dataset
train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

if __name__ == "__main__":

    logging.basicConfig(filename = "Transformer.log",
                        level = logging.DEBUG,
                        format = "%(Levelname)s %(asctime)s - %(message)s",
                        filemode = 'w')
    logger = logging.getLogger("TransformerLog")

    logger.info("Our first message.")    
    

