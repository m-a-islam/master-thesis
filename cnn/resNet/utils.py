from fvcore.nn import FlopCountAnalysis, parameter_count

def calculate_cost(model, input_tensor):
    # Calculate FLOPs and MACs
    flops = FlopCountAnalysis(model, input_tensor)
    macs = flops.total()
    
    # Calculate parameter size (model size)
    params = parameter_count(model)
    size_in_MB = sum(params.values()) * 4 / 1024 / 1024  # assuming 4 bytes per parameter

    return macs, size_in_MB

# Example input tensor for CIFAR-10 (3x32x32)
input_tensor = torch.randn(1, 3, 32, 32)
masked_resnet = MaskedResNet([2, 2, 2])

# Calculate cost for the masked model
macs, size = calculate_cost(masked_resnet, input_tensor)
print(f"MACs: {macs}, Model Size: {size:.2f} MB")
