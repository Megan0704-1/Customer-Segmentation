import onnx
import onnxruntime
import numpy as np
import pandas as pd

# Load the ONNX model
model_path = "onnx_models/resnet_simplified.onnx"
model = onnx.load(model_path)

# Helper function to get tensor shape
def get_tensor_shape(value_info):
    return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

# Create a dictionary to map tensor names to their shapes
tensor_shapes = {}
for value_info in model.graph.value_info:
    tensor_shapes[value_info.name] = get_tensor_shape(value_info)

# Add input and output tensors to the dictionary
for input in model.graph.input:
    tensor_shapes[input.name] = get_tensor_shape(input)
for output in model.graph.output:
    tensor_shapes[output.name] = get_tensor_shape(output)

# Iterate over layers and extract convolutional details
conv_layers = []
for node in model.graph.node:
    if node.op_type == "Conv":
        layer_name = node.name if node.name else f"Conv_{len(conv_layers)}"

        # Get input and output shapes
        input_shape = tensor_shapes.get(node.input[0], "Unknown")
        output_shape = tensor_shapes.get(node.output[0], "Unknown")

        # Extract weight tensor details
        weight_shape = None
        for initializer in model.graph.initializer:
            if initializer.name == node.input[1]:  # Find corresponding weight tensor
                weight_shape = list(initializer.dims)
                break

        conv_layers.append([layer_name, input_shape, output_shape, weight_shape])

# Print extracted information
df = pd.DataFrame(conv_layers, columns=["Layer Name", "Input Shape", "Output Shape", "Weight Shape"])
print(df)

latex_code = df.to_latex(index=False)

print(latex_code)

with open("table.tex", "w") as fh:
    fh.write(latex_code)


