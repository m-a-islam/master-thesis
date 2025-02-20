import json

# Load search space from JSON file
with open('search_space.json', 'r') as f:
    search_space = json.load(f)['search_space']

PRIMITIVES = {
    'CNN': search_space['CNN_operations'],
    'MLP': search_space['MLP_operations'],
    'Fusion': search_space['Fusion_operations']
}