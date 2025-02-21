# primitives.py
# Define the search space for different components
PRIMITIVES = {
    'general': [
        'none',                      # No connection
        'skip_connect',              # Identity (ResNet-style skip)
        'max_pool_2x2',             # Max Pooling (Downsampling)
        'avg_pool_2x2',             # Average Pooling
        'dropout'                    # Dropout Layer for Regularization
    ],
    'conv': [
        'conv_1x1',                 # Pointwise Convolution
        'conv_3x3',                 # Standard 3x3 Convolution
        'conv_5x5',                 # Larger 5x5 Convolution
        'conv_7x7',                 # Very Large 7x7 Convolution
        'depthwise_conv_3x3',       # Depthwise Separable Conv
        'dilated_conv_3x3',         # Dilated Convolution
        'grouped_conv_3x3',         # Grouped Convolution
    ],
    'norm': [
        'batch_norm',               # Batch Normalization Only
        'layer_norm',               # Layer Normalization
    ],
    'attention': [
        'squeeze_excitation',       # SE Layer for Attention
    ],
    'mlp': [
        'mlp',                      # Fully Connected MLP
    ]
}

# Flatten primitives for backward compatibility
ALL_PRIMITIVES = (
    PRIMITIVES['general'] + 
    PRIMITIVES['conv'] + 
    PRIMITIVES['norm'] + 
    PRIMITIVES['attention'] + 
    PRIMITIVES['mlp']
)