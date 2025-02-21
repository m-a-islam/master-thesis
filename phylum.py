import json
from typing import Dict, List

class SearchSpace:
    def __init__(self, config_path: str = 'search_space.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)['search_space']
        
        self.primitives = {
            'CNN': self.config['CNN_operations'],
            'MLP': self.config['MLP_operations'],
            'Fusion': self.config['Fusion_operations']
        }
        
        self.cells = self.config['cells']
    
    def get_operations(self, cell_type: str) -> List[str]:
        """Get available operations for a specific cell type."""
        return self.primitives.get(cell_type, [])
    
    def get_cell_config(self, cell_type: str) -> Dict:
        """Get cell configuration for a specific cell type."""
        return self.cells.get(cell_type, {})

# Create global instance
SEARCH_SPACE = SearchSpace()
PRIMITIVES = SEARCH_SPACE.primitives