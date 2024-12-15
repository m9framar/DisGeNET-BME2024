# data_prep.py
import requests
import json
import time
import pandas as pd
import os

def get_api_data(api_key):
    params = {
        "disease": """DO_1324, DO_1612, DO_10286, DO_9256, DO_10652,
        DO_14330, DO_1339, DO_1485, DO_12858, DO_11723, DO_9744,
        DO_9352, DO_1459, DO_9255, DO_7148, DO_9074, DO_1826, DO_332, 
        DO_13833, DO_12356, DO_5419, DO_3312, DO_1287, DO_1985, DO_12858"""
    }
    
    headers = {
        'Authorization': api_key,
        'accept': 'application/json'
    }
    
    all_payloads = []
    for page_number in range(100):
        params['page_number'] = page_number
        response = requests.get(
            "https://api.disgenet.com/api/v1/gda/summary",
            params=params, headers=headers, verify=False
        )
        
        if not response.ok and response.status_code == 429:
            while not response.ok:
                wait_time = int(response.headers['x-rate-limit-retry-after-seconds'])
                print(f"Rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                response = requests.get(
                    "https://api.disgenet.com/api/v1/gda/summary",
                    params=params, headers=headers, verify=False
                )
                if response.ok:
                    break
        
        response_parsed = json.loads(response.text)
        all_payloads.extend(response_parsed.get("payload", []))
    
    return pd.DataFrame(all_payloads)

def standardize_disease_names():
    return {
    "Alzheimer's dementia senile and presenile forms": 'Alzheimer Disease',
    "familial alzheimer's disease": 'Alzheimer Disease',
    'AMYOTROPHIC LATERAL SCLEROSIS-PARKINSONISM/DEMENTIA COMPLEX 1': 'Parkinson Disease',
    "Parkinson's Disease": 'Parkinson Disease',
    "Primary Parkinsonism or Parkinson's disease": 'Parkinson Disease',
    "Idiopathic Parkinsonism or Parkinson's disease": 'Parkinson Disease',
    "Huntington's Disease": 'Huntington Disease',
    'HUNTINGTON DIS JUVENILE ONSET': 'Huntington Disease',
    "Dementia in Huntington's disease": 'Huntington Disease',
    'Dystrophies, Muscular':'Duchenne Muscular Dystrophy',
    'MUSCULAR DYSTROPHY, BECKER TYPE':'Duchenne Muscular Dystrophy',
    'Anemia, Diamond Blackfan':'Sickle Cell Anemia',
    'Anemia, Congenital Hypoplastic':'Sickle Cell Anemia',
    'CARCINOMA OF BREAST':'Breast Cancer',
    'Malignant neoplasm of lower lobe, bronchus or lung':'Lung Cancer',
    'Malignant neoplasm of other parts of bronchus or lung':'Lung Cancer',
    'Malignant neoplasm of middle lobe, bronchus or lung':'Lung Cancer',
    'Malignant neoplasm of upper lobe, bronchus or lung':'Lung Cancer',
    'CARCINOMA OF PROSTATE':'Prostate Cancer',
    'SYSTEMIC LUPUS ERYTHEMATOSIS':'Systemic lupus erythematosus',
    'SYSTEMIC LUPUS ERYTHEMATOSUS 16':'Systemic lupus erythematosus',
    'Bipolar I disorder':'Bipolar Disorder',
    'Mixed bipolar disorder, NOS':'Bipolar Disorder',
    'Bipolar II disorder':'Bipolar Disorder',
    'diabetes (mellitus) due to autoimmune process':'Type 1 Diabetes',
    'diabetes (mellitus) due to immune mediated pancreatic islet beta-cell destruction':'Type 1 Diabetes',
    'Diabetes Mellitus, Insulin-Dependent':'Type 1 Diabetes',
    'ketosis prone diabetes':'Type 2 Diabetes',
    'idiopathic diabetes (mellitus)':'Type 2 Diabetes',
    'Insulin-resistant diabetes mellitus':'Type 2 Diabetes',
    'NIDDM1':'Type 2 Diabetes',
    'CRC': 'Colorectal Cancer',
    'Malignant neoplasm of colon': 'Colorectal Cancer',
    'Malignant neoplasm of large intestine': 'Colorectal Cancer',
    'Congenital hypothyroidism':'Hypothyroidism',
    'secondary hypothyroidism (diagnosis)':'Hypothyroidism',
    'Primary hypothyroidism':'Hypothyroidism',
    'central hypothyroidism':'Hypothyroidism',
    # Alzheimer's Disease
    "Alzheimer's Disease": "Alzheimer's Disease",
    "Familial Alzheimer's Disease": "Alzheimer's Disease",
    
    # Parkinson's Disease
    "Parkinson's Disease": "Parkinson's Disease",
    "Hemiparkinsonism": "Parkinson's Disease",
    "Primary Parkinsonism": "Parkinson's Disease",
    "Idiopathic Parkinsonism": "Parkinson's Disease",
    "Amyotrophic Lateral Sclerosis-Parkinsonism/Dementia Complex 1": "Parkinson's Disease",
    
    # Huntington's Disease
    "Huntington's Disease": "Huntington's Disease",
    "Chorea, Huntington": "Huntington's Disease",
    "Dementia in Huntington's Disease": "Huntington's Disease",
    
    # Duchenne Muscular Dystrophy
    "Duchenne Muscular Dystrophy": "Duchenne Muscular Dystrophy",
    "Duchenne Becker Muscular Dystrophy": "Duchenne Muscular Dystrophy",
    "Muscular Dystrophy, Duchenne": "Duchenne Muscular Dystrophy",
    
    # Sickle Cell Anemia
    "Sickle Cell Anemia": "Sickle Cell Anemia",
    
    # Cystic Fibrosis
    "Cystic Fibrosis": "Cystic Fibrosis",
    
    # Down Syndrome
    "Down Syndrome": "Down Syndrome",
    
    # Marfan Syndrome
    "Marfan Syndrome": "Marfan Syndrome",
    
    # Fabry Disease
    "Fabry Disease": "Fabry Disease",
    
    # Breast Cancer
    "Breast Neoplasms": "Breast Cancer",
    "Cancer, Breast": "Breast Cancer",
    "Carcinoma of Breast": "Breast Cancer",
    "Breast Cancer": "Breast Cancer",
    
    # Lung Cancer
    "Cancer, Lung": "Lung Cancer",
    "Malignant Neoplasm of Bronchus or Lung": "Lung Cancer",
    "Lung Cancer": "Lung Cancer",
    
    # Prostate Cancer
    "Carcinoma of Prostate": "Prostate Cancer",
    "Prostate Cancer": "Prostate Cancer",
    
    # Colorectal Cancer
    "Colorectal Neoplasm": "Colorectal Cancer",
    "Malignant Neoplasm of Colon": "Colorectal Cancer",
    "CRC (Colorectal Cancer)": "Colorectal Cancer",
    "Malignant Neoplasm of Large Intestine": "Colorectal Cancer",
    "Colorectal Cancer": "Colorectal Cancer",
    
    # Type 1 Diabetes
    "Type 1 Diabetes": "Type 1 Diabetes",
    
    # Type 2 Diabetes
    "Adult-Onset Diabetes Mellitus": "Type 2 Diabetes",
    "Insulin-Resistant Diabetes Mellitus": "Type 2 Diabetes",
    "Type 2 Diabetes": "Type 2 Diabetes",
    
    # Hemophilia
    "Hemophilia": "Hemophilia",
    
    # Polycystic Kidney Disease
    "Polycystic Kidney Disease": "Polycystic Kidney Disease",
    
    # Systemic Lupus Erythematosus
    "Lupus": "Systemic Lupus Erythematosus",
    "Systemic Lupus Erythematosus 16": "Systemic Lupus Erythematosus",
    
    # Rheumatoid Arthritis
    "Rheumatoid Arthritis": "Rheumatoid Arthritis",
    
    # Autoimmune Diseases
    "Autoimmune Diseases (e.g., Rheumatoid Arthritis)": "Autoimmune Diseases",
    
    # Schizophrenia
    "Schizophrenia": "Schizophrenia",
    
    # Bipolar Disorder
    "Bipolar Disorders": "Bipolar Disorder",
    "Bipolar I Disorder": "Bipolar Disorder",
    "Bipolar II Disorder": "Bipolar Disorder",
    "Bipolar Depression": "Bipolar Disorder",
    "Mixed Bipolar Disorder": "Bipolar Disorder",
    "NOS": "Bipolar Disorder",
    
    # Tourette Syndrome
    "Tourette Syndrome": "Tourette Syndrome",
    
    # Porphyria
    "Porphyria": "Porphyria",
    
    # Amyotrophic Lateral Sclerosis (ALS)
    "Amyotrophic Lateral Sclerosis": "Amyotrophic Lateral Sclerosis (ALS)",
    
    # Cardiovascular Diseases
    "Cardiovascular Disease": "Cardiovascular Diseases",
    "Cardiac Disease": "Cardiovascular Diseases",
    "Cardiovascular Diseases": "Cardiovascular Diseases"
    }

def process_data(df):
    # Filter columns
    df_filtered = df[['geneNcbiID', 'geneDSI', 'geneDPI', 'diseaseName', 'score']]
    
    # Standardize disease names
    replacements = standardize_disease_names()
    df_filtered['diseaseName'] = df_filtered['diseaseName'].replace(replacements)
    
    return df_filtered

def main():
    api_key = os.getenv('DISGENET_API_KEY')
    if not api_key:
        raise ValueError("DISGENET_API_KEY environment variable not set")
    
    # Get data from API
    print("Fetching data from DisGeNET API...")
    df = get_api_data(api_key)
    df.to_csv("disgenet_gda_summary.csv", index=False)
    
    # Process data
    print("Processing data...")
    df_filtered = process_data(df)
    
    # Save filtered data to mounted volume
    output_path = os.path.join('data', 'finalized_data.csv')
    print(f"Saving processed data to {output_path}...")
    df_filtered.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()