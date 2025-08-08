import torch
import json
import os
from typing import List, Tuple, Optional, Dict, Any
from torch_geometric.data import Data, Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import random_split


class VulnerabilityDataset(Dataset):
    """Dataset class for vulnerability detection graphs."""
    
    def __init__(self, 
                 data_dir: str,
                 transform=None,
                 pre_transform=None,
                 max_nodes: int = 1000):
        self.data_dir = data_dir
        self.max_nodes = max_nodes
        self.graphs = []
        self.labels = []
        
        super(VulnerabilityDataset, self).__init__(data_dir, transform, pre_transform)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load graph data from JSON files."""
        print(f"Loading data from {self.data_dir}")
        
        json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            file_path = os.path.join(self.data_dir, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract graph information
                graph_data = self._parse_json_to_graph(data, json_file)
                if graph_data is not None:
                    self.graphs.append(graph_data)
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Loaded {len(self.graphs)} graphs")
    
    def _parse_json_to_graph(self, data: Dict[str, Any], filename: str) -> Optional[Data]:
        """Parse JSON data to PyTorch Geometric graph."""
        try:
            # Parse CFG (Control Flow Graph) from the actual JSON structure
            if 'cfg' not in data:
                return None
            
            cfg_nodes = data['cfg']
            
            # Limit number of nodes
            if len(cfg_nodes) > self.max_nodes:
                return None
            
            # Create node features from CFG nodes
            node_features = []
            for node in cfg_nodes:
                feature = self._extract_node_features_from_cfg(node)
                node_features.append(feature)
            
            # Create edges based on CFG structure (simple sequential flow for now)
            edge_indices = []
            for i in range(len(cfg_nodes) - 1):
                edge_indices.append([i, i + 1])
            
            # If we have AST data, we can add more sophisticated edges
            if 'ast' in data:
                ast_edges = self._extract_ast_edges(data['ast'], len(cfg_nodes))
                edge_indices.extend(ast_edges)
            
            # Need at least one edge for a valid graph
            if not edge_indices:
                # Create self-loop if no edges
                if len(cfg_nodes) > 0:
                    edge_indices.append([0, 0])
                else:
                    return None
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            # Determine label from filename
            label = self._extract_label_from_filename(filename)
            y = torch.tensor([label], dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index, y=y)
        
        except Exception as e:
            print(f"Error parsing graph from {filename}: {e}")
            return None
    
    def _extract_node_features_from_cfg(self, cfg_node: Dict[str, Any]) -> List[float]:
        """Extract features from a CFG node."""
        features = [0.0] * 128  # Default feature size
        
        # Extract basic features from CFG node
        if 'id' in cfg_node:
            # Node ID (normalized)
            features[0] = float(cfg_node['id']) / 100.0
        
        if 'statements' in cfg_node:
            # Number of statements in this CFG node
            num_statements = len(cfg_node['statements'])
            features[1] = min(num_statements / 10.0, 1.0)  # Normalize to [0,1]
            
            # Simple statement type features (placeholder)
            for i, stmt in enumerate(cfg_node['statements'][:10]):  # Limit to first 10
                if isinstance(stmt, dict) and 'kind' in stmt:
                    # Hash the statement kind to a feature index
                    feature_idx = (hash(stmt['kind']) % 20) + 2
                    if feature_idx < 128:
                        features[feature_idx] = 1.0
        
        return features
    
    def _extract_ast_edges(self, ast_data: Dict[str, Any], num_cfg_nodes: int) -> List[List[int]]:
        """Extract additional edges from AST structure."""
        edges = []
        
        # This is a simplified approach - in practice you'd want more sophisticated AST->CFG mapping
        # For now, just add some random edges based on AST structure
        if 'body' in ast_data and isinstance(ast_data['body'], dict):
            if 'children' in ast_data['body']:
                children = ast_data['body']['children']
                if len(children) > 1 and num_cfg_nodes > 1:
                    # Connect first and last CFG nodes if we have AST children
                    edges.append([0, num_cfg_nodes - 1])
        
        return edges
    
    def _extract_node_features(self, node: Dict[str, Any]) -> List[float]:
        """Extract features from a node (legacy method for compatibility)."""
        # Placeholder implementation - customize based on your node structure
        features = [0.0] * 128  # Default feature size
        
        # Example feature extraction
        if 'type' in node:
            # Encode node type
            node_type = str(node['type'])
            features[0] = hash(node_type) % 100 / 100.0
        
        if 'value' in node:
            # Encode node value
            try:
                value = float(node['value'])
                features[1] = value
            except:
                features[1] = 0.0
        
        # Add more feature extraction logic here
        
        return features
    
    def _extract_label_from_filename(self, filename: str) -> int:
        """Extract vulnerability label from filename."""
        # Example logic - adjust based on your naming convention
        if 'good' in filename.lower() or 'safe' in filename.lower():
            return 0  # Not vulnerable
        elif 'bad' in filename.lower() or 'vuln' in filename.lower():
            return 1  # Vulnerable
        else:
            # Default based on CWE patterns
            return 1 if 'cwe' in filename.lower() else 0
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        return self.graphs[idx]


def create_data_splits(dataset: VulnerabilityDataset, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train, validation, and test sets."""
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return random_split(dataset, [train_size, val_size, test_size])
