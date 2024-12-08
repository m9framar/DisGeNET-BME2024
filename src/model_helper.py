from datetime import datetime
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from sklearn.metrics import roc_auc_score
import tqdm
import os
def get_model_path(filename):
    """Get full path for model file in results directory"""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, filename)
def prepare_hetero_data(df):
    # Prepare gene node features using gene_index as indices
    genes = df[['gene_index', 'geneDSI', 'geneDPI']].drop_duplicates().set_index('gene_index')
    num_genes = genes.index.max() + 1  # Assuming indices start from 0
    gene_features = torch.zeros((num_genes, 2), dtype=torch.float)
    gene_features[genes.index] = torch.tensor(genes[['geneDSI', 'geneDPI']].values, dtype=torch.float)

    # Prepare disease nodes
    diseases = df[['disease_index']].drop_duplicates().set_index('disease_index')
    num_diseases = diseases.index.max() + 1  # Assuming indices start from 0

    # Prepare edge indices using existing indices
    edge_index = torch.tensor([
        df['gene_index'].values,
        df['disease_index'].values
    ], dtype=torch.long)

    # Edge attributes (scores)
    edge_attr = torch.tensor(df['score'].values, dtype=torch.float).unsqueeze(1)

    # Create HeteroData object
    data = HeteroData()

    # Add gene node features
    data['gene'].x = gene_features

    # Set the number of disease nodes (no features)
    data['disease'].num_nodes = num_diseases

    # Add edges between genes and diseases with edge attributes
    data['gene', 'associates_with', 'disease'].edge_index = edge_index
    data['gene', 'associates_with', 'disease'].edge_attr = edge_attr

    return data

def split_data(data):
    # Add reverse edges to allow message passing in both directions
    # Convert the graph to undirected (adds reverse edges)
    data = T.ToUndirected()(data)

    # Define the edge types for splitting
    edge_types = ('gene', 'associates_with', 'disease')
    rev_edge_types = ('disease', 'rev_associates_with', 'gene')  # Reverse edge type

    # Perform RandomLinkSplit with corrected parameter
    transform = T.RandomLinkSplit(
        num_val=0.1,                     # 10% for validation
        num_test=0.1,                    # 10% for testing
        disjoint_train_ratio=0.3,        # 30% of training edges for supervision
        neg_sampling_ratio=2.0,          # Negative edge ratio for evaluation
        is_undirected=True,              # Graph is undirected
        add_negative_train_samples=False,  # Negative edges generated on-the-fly during training
        edge_types=edge_types,
        rev_edge_types=rev_edge_types,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data

from torch_geometric.loader import LinkNeighborLoader

def create_data_loaders(train_data, val_data, test_data, edge_label_index, batch_size=128, num_neighbors=[10, 5], neg_sampling_ratio=2.0):
    # Training data loader
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=(
            edge_label_index,
            train_data[edge_label_index].edge_label_index
        ),
        edge_label=train_data[edge_label_index].edge_label,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True
    )

    # Validation data loader
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=(
            edge_label_index,
            val_data[edge_label_index].edge_label_index
        ),
        edge_label=val_data[edge_label_index].edge_label,
        shuffle=False
    )

    # Test data loader
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        edge_label_index=(
            edge_label_index,
            test_data[edge_label_index].edge_label_index
        ),
        edge_label=test_data[edge_label_index].edge_label,
        shuffle=False
    )

    return train_loader, val_loader, test_loader
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x_gene, x_disease, edge_label_index):
        row, col = edge_label_index
        x = torch.cat([x_gene[row], x_disease[col]], dim=-1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x.squeeze()

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_diseases, data_metadata):
        super().__init__()
        self.gene_lin = torch.nn.Linear(2, hidden_channels)
        self.disease_emb = torch.nn.Embedding(num_diseases, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, data_metadata)
        self.predictor = LinkPredictor(hidden_channels)

    def forward(self, data):
        x_dict = {
            'gene': self.gene_lin(data['gene'].x),
            'disease': self.disease_emb(torch.arange(data['disease'].num_nodes).to(data['gene'].x.device))
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return self.predictor(
            x_dict['gene'],
            x_dict['disease'],
            data['gene', 'associates_with', 'disease'].edge_label_index,
        )

def create_model(hidden_channels, num_diseases, data_metadata):
    return Model(hidden_channels, num_diseases, data_metadata)
@torch.no_grad()
def evaluate(loader, model, device):
    """Evaluate model on given loader"""
    model.eval()
    preds, labels = [], []
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        pred = model(batch)
        preds.append(pred.sigmoid().cpu())
        labels.append(batch['gene', 'associates_with', 'disease'].edge_label.cpu())
    return roc_auc_score(
        torch.cat(labels).numpy(),
        torch.cat(preds).numpy()
    )


def train_model(model, train_loader, val_loader, test_loader, num_epochs=100, patience=10, device='cuda'):
    """Train model with early stopping and compare against best model"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    counter = 0
    train_losses, val_aucs, test_aucs = [], [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            target = batch['gene', 'associates_with', 'disease'].edge_label.float()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.size(0)
            num_batches += 1
        
        avg_loss = total_loss / (num_batches * train_loader.batch_size)
        train_losses.append(avg_loss)
        
        # Evaluate
        with torch.no_grad():
            model.eval()
            val_auc = evaluate(val_loader, model, device)
            test_auc = evaluate(test_loader, model, device)
            val_aucs.append(val_auc)
            test_aucs.append(test_auc)
        
        print(f'Epoch {epoch:03d}:')
        print(f'Train Loss: {avg_loss:.4f}')
        print(f'Val AUC: {val_auc:.4f}')
        print(f'Test AUC: {test_auc:.4f}')
        
        # Early stopping with current model save
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), get_model_path('current_model.pt'))
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping!')
                break
    
    # Load early-stopped model and evaluate
    # Load early-stopped model and evaluate
    model.load_state_dict(torch.load(get_model_path('current_model.pt'), weights_only=True))
    current_test_auc = evaluate(test_loader, model, device)

    # Compare with existing best model if it exists
    if os.path.exists(get_model_path('best_model.pt')):
        # Load best model and evaluate
        model.load_state_dict(torch.load(get_model_path('best_model.pt'), weights_only=True))
        best_test_auc = evaluate(test_loader, model, device)
        
        if current_test_auc > best_test_auc:
            print(f"New best model found: {current_test_auc:.4f} > {best_test_auc:.4f}")
            # Simply copy the current model file to best model
            import shutil
            shutil.copy2(get_model_path('current_model.pt'), get_model_path('best_model.pt'))
            # Load current model state back
            model.load_state_dict(torch.load(get_model_path('current_model.pt'), weights_only=True))
        else:
            print(f"Keeping previous model: {best_test_auc:.4f} > {current_test_auc:.4f}")
            # Keep best model loaded
    else:
        print("First run - saving as best model")
        # Simply copy the current model file
        import shutil
        shutil.copy2(get_model_path('current_model.pt'), get_model_path('best_model.pt'))
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'test_aucs': test_aucs,
        'final_test_auc': current_test_auc
    }
def get_and_save_novel_predictions(model, data, mapping,device, batch_size=1024, output_file_prefix='novel_predictions'):
    import pandas as pd
    model.eval()
    
    # Get ALL known edges from original data
    known_edges = set((int(i), int(j)) for i, j in data['gene', 'associates_with', 'disease'].edge_index.t().tolist())
    print(f"Known edges in original data: {len(known_edges)}")
    
    # Debug: Print some known edges
    print("Sample known edges:", list(known_edges)[:5])
    
    # Add timestamp to output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"{output_file_prefix}_{timestamp}.csv"
    
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['gene_idx', 'disease_idx', 'gene_ncbi', 'disease_name', 'prediction_score']).to_csv(output_file, index=False)
    
    try:
        # Generate novel pairs with explicit type conversion
        novel_pairs = []
        for g in range(data['gene'].num_nodes):
            for d in range(data['disease'].num_nodes):
                if (int(g), int(d)) not in known_edges:
                    novel_pairs.append([g, d])
        
        print(f"Novel pairs to evaluate: {len(novel_pairs)}")
        # Debug: Print some novel pairs
        print("Sample novel pairs:", novel_pairs[:5])
        
        # Process in batches
        for i in tqdm.tqdm(range(0, len(novel_pairs), batch_size)):
            batch_pairs = novel_pairs[i:i+batch_size]
            edge_label_index = torch.tensor(batch_pairs, dtype=torch.long).t().to(device)
            
            batch_data = data.clone()
            batch_data['gene', 'associates_with', 'disease'].edge_label_index = edge_label_index
            batch_data = batch_data.to(device)
            
            with torch.no_grad():
                batch_pred = model(batch_data)
                pred_probs = batch_pred.sigmoid().cpu().numpy()
            
            batch_results = []
            for j, (pair, prob) in enumerate(zip(batch_pairs, pred_probs)):
                if prob >= 0.5:
                    batch_results.append({
                        'gene_idx': int(pair[0]),
                        'disease_idx': int(pair[1]),
                        'gene_ncbi': mapping['gene'][1][pair[0]],
                        'disease_name': mapping['disease'][1][pair[1]],
                        'prediction_score': float(prob)
                    })
            
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                batch_df = batch_df.sort_values(by='prediction_score', ascending=False)
                batch_df.to_csv(output_file, mode='a', header=False, index=False)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise
    
    # Verify results with explicit type conversion
    if os.path.exists(output_file):
        final_df = pd.read_csv(output_file)
        edge_pairs = set((int(row.gene_idx), int(row.disease_idx)) for _, row in final_df.iterrows())
        
        # Debug: Print some predicted edges
        print("Sample predicted edges:", list(edge_pairs)[:5])
        
        overlap = edge_pairs.intersection(known_edges)
        if overlap:
            print(f"Warning: Found {len(overlap)} predictions that were in training data!")
            print("Sample overlapping edges:", list(overlap)[:5])
        
        return final_df.nlargest(10, 'prediction_score')
    return None