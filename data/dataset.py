import os
import json
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from pathlib import Path
from typing import List, Dict

class ASTCFGDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        self.root_dir = root
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        data_list = []
        node_kinds = set()
        edge_types = {"AST", "CFG"}

        # First pass to build vocab of node kinds
        for json_file in Path(self.root_dir).rglob("*.json"):
            with open(json_file) as f:
                js = json.load(f)
                self._collect_ast_kinds(js["ast"]["body"], node_kinds)
                for block in js["cfg"]:
                    node_kinds.update(block["statements"])

        self.kind_vocab = {k: i for i, k in enumerate(sorted(node_kinds))}
        self.edge_vocab = {"AST": 0, "CFG": 1}

        for json_file in Path(self.root_dir).rglob("*.json"):
            try:
                with open(json_file) as f:
                    js = json.load(f)
                    data = self._json_to_data(js)
                    data_list.append(data)
            except Exception as e:
                print(f"Failed to process {json_file}: {e}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _collect_ast_kinds(self, node, kind_set):
        if not node or "kind" not in node:
            return
        kind_set.add(node["kind"])
        for child in node.get("children", []):
            self._collect_ast_kinds(child, kind_set)

    def _extract_ast(self, node, edges, node_list) -> int:
        """
        Recursively flatten AST into nodes and edges.
        Returns the index of the current node.
        """
        if not node or "kind" not in node:
            return None

        cur_idx = len(node_list)
        node_list.append(node["kind"])
        for child in node.get("children", []):
            child_idx = self._extract_ast(child, edges, node_list)
            if child_idx is not None:
                edges.append((cur_idx, child_idx, "AST"))
        return cur_idx

    def _json_to_data(self, js) -> Data:
        ast_nodes = []
        edges = []

        # Parse AST
        self._extract_ast(js["ast"]["body"], edges, ast_nodes)

        # Add CFG nodes (if not already added)
        cfg_offset = len(ast_nodes)
        for block in js["cfg"]:
            for stmt in block["statements"]:
                ast_nodes.append(stmt)  # treat statement kind as a node

        # Add CFG edges (block-wise sequential)
        for i in range(len(js["cfg"]) - 1):
            s = cfg_offset + i
            t = cfg_offset + i + 1
            edges.append((s, t, "CFG"))

        # Create tensors
        x = torch.tensor([self.kind_vocab[k] for k in ast_nodes], dtype=torch.long).unsqueeze(1)

        edge_index = []
        edge_attr = []

        for src, tgt, etype in edges:
            edge_index.append([src, tgt])
            edge_attr.append(self.edge_vocab[etype])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).unsqueeze(1)

        label = torch.tensor([js.get("label", 0)], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=label)

def get_ast_cfg_loader(data_dir: str, batch_size=32, shuffle=True):
    dataset = ASTCFGDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
