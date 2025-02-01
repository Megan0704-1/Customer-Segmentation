import onnx
from onnxsim import simplify
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output-path",
    type=str,
    required=True,
    help="Path to save the simplified onnx model",
)

parser.add_argument(
    "--input-path",
    type=str,
    required=True,
    help="Path to the input onnx model",
)

parser.add_argument(
    "--model-type",
    type=str,
    default="mlp",
    help="Choice which model to use",
)

args = parser.parse_args()

####################################################################
# switch models
print(f"Model type to visualize: {args.model_type}")

if args.model_type == "mlp":
    from part1.part1_v1 import SimpleMLP

    model = SimpleMLP()
elif args.model_type == "cnn":
    from part2 import SimpleCNN

    model = SimpleCNN()
else:
    print(f"Unrecognize model type")
    exit(1)
####################################################################
# Create a random input for the onnnx export script

batch_size = 32

data = torch.rand(batch_size, 3, 28, 28)

torch.onnx.export(
    model,  # model being run
    data,  # model input (or a tuple for multiple inputs)
    args.input_path,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["ip"],  # the model's input names
    output_names=["op"],  # the model's output names
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
)

####################################################################
# Simplify the model using onnxsim

input_model_path = args.input_path  # Replace with your model's path
output_model_path = args.output_path

# Load the ONNX model
model = onnx.load(input_model_path)

# Simplify the model (includes constant propagation)
simplified_model, check = simplify(model)

# Check if simplification was successful
if check:
    print("Model simplification successful. Saving the simplified model...")
    # Save the simplified model
    onnx.save(simplified_model, output_model_path)
    print(f"Simplified model saved at: {output_model_path}")
else:
    print("Model simplification failed.")
