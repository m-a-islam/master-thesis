from collections import namedtuple
import json

# Load search space from a config file
def load_search_space(config_path='search_space.json'):
    with open(config_path, 'r') as f:
        search_space = json.load(f)['search_space']
    return {
        'CNN': search_space['CNN_operations'],
        'MLP': search_space['MLP_operations'],
        'Fusion': search_space['Fusion_operations']
    }

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
PRIMITIVES = load_search_space()

# Example predefined architectures
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2