import torch
import pandas as pd

def save_training_results(train_losses, val_aucs, test_aucs, output_dir='results'):
    """Save training metrics and plots for docker environment"""
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Training Progress Plot
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    
    # AUC plot
    plt.subplot(1, 2, 2)
    plt.plot(val_aucs, label='Validation AUC')
    plt.plot(test_aucs, label='Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores over Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'))
    plt.close()
    
    # 2. Save metrics as numpy file
    metrics_dict = {
        'train_loss': train_losses,
        'val_auc': val_aucs,
        'test_auc': test_aucs
    }
    np.save(os.path.join(output_dir, 'training_metrics.npy'), metrics_dict)
    
    # 3. Save metrics as text for easy reading
    with open(os.path.join(output_dir, 'final_metrics.txt'), 'w') as f:
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Validation AUC: {val_aucs[-1]:.4f}\n")
        f.write(f"Final Test AUC: {test_aucs[-1]:.4f}\n")

def create_mappings(df):
    """Create gene and disease index mappings from DataFrame"""
    gene_to_idx = dict(zip(df['geneNcbiID'].unique(), range(len(df['geneNcbiID'].unique()))))
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    disease_to_idx = dict(zip(df['diseaseName'].unique(), range(len(df['diseaseName'].unique()))))
    idx_to_disease = {v: k for k, v in disease_to_idx.items()}
    
    return gene_to_idx, idx_to_gene, disease_to_idx, idx_to_disease

@torch.no_grad()
def get_readable_predictions(model, data, mapping, device='cuda'):
    """Get predictions with gene and disease names from DataFrame"""
    try:

        
        # Setup model
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        data = data.to(device)
        
        # Get predictions
        pred = model(data)
        pred = pred.sigmoid().cpu().numpy()
        
        # Get edge indices
        edge_index = data['gene', 'associates_with', 'disease'].edge_label_index.cpu().numpy()
        
        # Create DataFrame with predictions and names
        df_pred = pd.DataFrame({
            'gene_idx': edge_index[0],
            'disease_idx': edge_index[1],
            'gene_ncbi': [mapping['gene'][1][idx] for idx in edge_index[0]],
            'disease_name': [mapping['disease'][1][idx] for idx in edge_index[1]],
            'prediction_score': pred
        })
        
        return df_pred
    
    except KeyError as e:
        print(f"Missing required column in DataFrame: {e}")
        raise
    except Exception as e:
        print(f"Error in prediction generation: {str(e)}")
        raise
def evaluate_link_prediction(predictions_df, test_data, output_dir='results', threshold=0.5):
    """Evaluate binary link prediction performance"""
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # All edges in test set are positive examples (they exist)
    # Get true labels from test data
    y_true = test_data['gene', 'associates_with', 'disease'].edge_label.cpu().numpy()
    y_pred = (predictions_df['prediction_score'] >= threshold).astype(int)
    y_scores = predictions_df['prediction_score']
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction Score Distribution
    sns.histplot(data=predictions_df, x='prediction_score', bins=50, ax=ax1)
    ax1.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax1.set_title('Distribution of Edge Existence Predictions')
    ax1.set_xlabel('Prediction Score (Existence Probability)')
    ax1.legend()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend()
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Disease-wise Performance (Fixed)
    disease_perf = predictions_df.groupby('disease_name')['prediction_score'].agg(['mean', 'std']).reset_index()
    disease_perf = disease_perf.sort_values('mean')
    
    # Create basic barplot
    sns.barplot(data=disease_perf, x='mean', y='disease_name', ax=ax4)
    
    # Add error bars manually
    ax4.errorbar(x=disease_perf['mean'], y=range(len(disease_perf)),
                xerr=disease_perf['std'], fmt='none', color='black', capsize=5)
    
    ax4.set_title('Disease-wise Prediction Confidence')
    ax4.set_xlabel('Mean Prediction Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'))
    plt.close()
    
    # Save classification report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred))
    
    # Save threshold analysis to file
    with open(os.path.join(output_dir, 'threshold_analysis.txt'), 'w') as f:
        f.write("\nPrediction Score Distribution:\n")
        for t in [0.3, 0.5, 0.7, 0.9]:
            correct = (predictions_df['prediction_score'] >= t).sum()
            total = len(predictions_df)
            f.write(f"Predictions >= {t}: {correct} ({correct/total*100:.1f}%)\n")
    
    return y_true, y_pred, y_scores