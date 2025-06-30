import torch
from torch_geometric.loader import DataLoader
from dataset import ASTCFGDataset

def main():
    data_dir = "/app/data/outputs"
    batch_size = 1

    print(f"Loading dataset from: {data_dir}")
    dataset = ASTCFGDataset(data_dir)
    print(f"Total graphs: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, batch in enumerate(loader):
        print("\nBatch", i)
        print(f"x.shape (node features):     {batch.x.shape}")
        print(f"edge_index.shape:           {batch.edge_index.shape}")
        print(f"edge_attr.shape:            {batch.edge_attr.shape}")
        print(f"Labels (y):                 {batch.y.tolist()}")
        print(f"Number of graphs in batch:  {batch.num_graphs}")
        break  # Only print the first batch

if __name__ == "__main__":
    main()
