import torch
from pit_mc_darts import PITDARTS

def load_pit_model(saved_model_path):
    model = PITDARTS(init_channels=8, num_classes=10, layers=4).to('cpu')
    checkpoint = torch.load(saved_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    return model

def convert_to_onnx(model, output_path="pit_model.onnx"):
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST input shape
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=True
    )
    print(f"✅ Model saved as ONNX to {output_path}")

def main():
    saved_pit_model_path = "trained-models/mnist_pit_normal_cnn.pth"
    pit_model = load_pit_model(saved_pit_model_path)
    path = "trained-models/onnx/mnist_pit_normal_cnn.onnx"
    convert_to_onnx(pit_model, path)

if __name__ == "__main__":
    main()