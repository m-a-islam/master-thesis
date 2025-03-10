import json
from typing import Dict, List

class SearchSpace:
    def __init__(self, config_path: str = 'search_space.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['search_space']
        
        self.primitives = {
            'CNN': self.config['CNN_operations'],
        }
        
        self.cells = self.config['cells']
    
    def get_operations(self, cell_type: str) -> List[str]:
        return self.primitives.get(cell_type, [])
    
    def get_cell_config(self, cell_type: str) -> Dict:
        return self.cells.get(cell_type, {})

SEARCH_SPACE = SearchSpace()
PRIMITIVES = SEARCH_SPACE.primitives