import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from model_helper import prepare_hetero_data,split_data, create_data_loaders, create_model, train_model,get_and_save_novel_predictions, get_model_path
import argparse
from plots import save_training_results, create_mappings, get_readable_predictions, evaluate_link_prediction


def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    df['score'] = df['score'].round(2)
    df = df.drop_duplicates()
    gene_to_idx, idx_to_gene, disease_to_idx, idx_to_disease= create_mappings(df)
    mapping={
        'gene': (gene_to_idx, idx_to_gene),
        'disease': (disease_to_idx, idx_to_disease)
    }
    # Create initial data
    data = prepare_hetero_data(df)
    
    # Apply transform and store result
    data = T.ToUndirected()(data)
    print("Data metadata after transform:", data.metadata())
    
    # Split data
    train_data, val_data, test_data = split_data(data)
    
    return data, train_data, val_data, test_data, mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with the given CSV file.')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file', default='finalized_data.csv')
    args = parser.parse_args()
    
    # Get transformed data
    data, train_data, val_data, test_data, mapping = prepare_data(args.csv_file)
    
    # Define edge type for link prediction
    edge_label_index = ('gene', 'associates_with', 'disease')
    
    # Create data loaders using transformed data
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, edge_label_index
    )
    
    # Create model with transformed data metadata
    model = create_model(hidden_channels=64, num_diseases=23, data_metadata=data.metadata())
    res= train_model(model, train_loader, val_loader, test_loader, num_epochs=100)
    save_training_results(res['train_losses'], res['val_aucs'], res['test_aucs'], output_dir='results')
    model.load_state_dict(torch.load(get_model_path('best_model.pt'), weights_only=True))
    
    predictions_df = get_readable_predictions(model, test_data, mapping, device='cpu')
    evaluate_link_prediction(predictions_df, test_data, threshold=0.5)
    # Execute with original data instead of test_data
    print("Starting prediction generation...")
    top_predictions = get_and_save_novel_predictions(model, data, mapping, 'cpu')
    print("\nTop 10 Novel Predictions (score >= 0.5):")
    print(top_predictions[['gene_ncbi', 'disease_name', 'prediction_score']])
