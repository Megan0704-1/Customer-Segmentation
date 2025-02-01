import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import onnx
from onnxsim import simplify
from torchvision import models, transforms
import time
import argparse

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} for inference")

####################################################################
# Download a pretrained model for resnet18
def main():
    resnet18 = models.resnet18(pretrained=True)
    utils = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_convnets_processing_utils"
    )

    resnet18.eval().to(device)

####################################################################
# Download images and batch them up for inference. You can either add
# image urls below to increase the batch size or you have images from
# other sources

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and testing (default: 64)",
    )
    args = parser.parse_args()

    batch_size = args.batch_size

    uris = [
        "http://images.cocodataset.org/test-stuff2017/000000024309.jpg",
    ]

    x_uri_bs = [uris[0] for i in range(batch_size)]

    batch = torch.cat([utils.prepare_input_from_uri(uri) for uri in x_uri_bs]).to(device)

    total_time = 0
    with torch.no_grad():
        # Run 100 time to remove any measurement/system overhead
        start_time = time.time()
        for i in range(100):
            output = torch.nn.functional.softmax(resnet18(batch), dim=1)
        end_time = time.time()

        total_time = (end_time - start_time) / 100

    print("Inference time for batchsize {} is {}".format(batch_size, total_time))

# Uncomment the below line if you want to see the prediction output
    results = utils.pick_n_best(predictions=output, n=5)

if __name__ == "__main__":
    main()
