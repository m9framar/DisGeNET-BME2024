## Team Name
      Shrekkentésch
## Project Members

- **Frank Marcell** - UMWAFS
- **Bindics Boldizsár** - Q12CTX
- **Smuk András** - D7S63U

## Project Goal

The goal of this project is to create a graph neural network for predicting disease-gene associations. Working with DisGeNET, a comprehensive database of these associations, we apply deep learning to an important challenge of bioinformatics.

## Dataset

[DisGeNET](https://www.disgenet.org/)

### The DISGENET schema

DISGENET data is standardized using community-supported standards and ontologies. The data can be filtered and selected by using different attributes, metrics and scores available for the genes, variants, diseases and their associations.
![Disgenet Schema](https://disgenet.com/static/images/Release_DGNplusSchema_SVG.svg)

## Related GitHub Repository

[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

## Related Papers

- [Paper 1](https://arxiv.org/abs/1607.00653)
- [Paper 2](https://arxiv.org/abs/1611.07308)

## Usage

### Training and Evaluation

To train and evaluate the model:

1. Build the Docker image:
      ```bash
      docker build -t gene-disease-predictor .
      ```

2. Run and save results:
      ```bash
      docker run -v ${PWD}/results:/app/results gene-disease-predictor
      ```

This will train the model, evaluate it, then generate training metrics/plots, along with predictions for the dataset.

### Data Generation

For generating and cleaning the DisGeNET data:

1. Build the Docker image:
      ```bash
      docker build -t data-prep -f data_prep.Dockerfile .
      ```

2. Run to call the API, scrape, and clean results:
      ```bash
      docker run -e DISGENET_API_KEY=[YOUR_KEY] -v ${PWD}/data:/app/data data-prep
      ```

This will create `finalized_data.csv` in the `data` folder of the project.