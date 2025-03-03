# Define the search space for different components
PRIMITIVES = {
    'general': [
        'skip_connect',
        'max_pool_2x2',
        'avg_pool_2x2',
    ],
    'conv': [
        'conv_1x1',
        'conv_3x3',
        'conv_5x5',
        'conv_7x7',
        'depthwise_conv_3x3',
    ],
    'norm': [
        'batch_norm',
    ],
}

ALL_PRIMITIVES = (
    PRIMITIVES['general'] +
    PRIMITIVES['conv'] +
    PRIMITIVES['norm']
)