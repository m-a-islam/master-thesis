PRIMITIVES = {
    'general': [
        'skip_connect',
        'max_pool_3x3'
    ],
    'conv': [
        'conv_1x1',
        'conv_3x3'
    ]
}

ALL_PRIMITIVES = (
    PRIMITIVES['general'] +
    PRIMITIVES['conv']
)