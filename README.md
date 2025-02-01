## Models:
- part1: MLP
- part2: CNN
- part3: Resnet (loaded from torchvision)
- part4: BERT


## Visualization
To visualize the model, run the visualization.py script
```bash
MODEL=resnet
python visualize.py --model-type $MODEL --input-path onnx_models/$MODEL_raw.onnx --output-path onnx_models/$MODEL_simplified.onnx
```
