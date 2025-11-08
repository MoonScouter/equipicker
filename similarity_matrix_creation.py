

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import time


# Read csv file and assign to variable
df_stock = pd.read_csv('us_stock_data.csv')
df_sector = pd.read_csv('pmi_sectors.csv')
# Placeholder for another csv containing sectors from PMI

# Constants
BATCH_SIZE = 100
MODEL_NAME = 'bert-base-uncased'

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


def preprocess_and_concatenate(df, type='stock'):

        if type == 'stock':
            # Remove rows where MarketCapCategory is 'Unknown'
            df.drop(df[df['MarketCapCategory'] == 'Unknown'].index, inplace=True)

            # Concatenate the text fields into a single text representation
            df['combined_text'] = df['GicSector'] + ' ' + df['GicIndustry'] + ' ' + df['GicSubIndustry'] + ' ' + df['Description']
            df['combined_text_simple'] = df['GicSector'] + ' ' + df['GicIndustry'] + ' ' + df['GicSubIndustry']

        elif type == 'sector':
            df['Details'] = df['Details'].fillna('')
            df['combined_text'] = df['Sector'] + ' ' + df['Details'] + ' ' + df['Type']

        return df


def batch_compute_embeddings(df):
    # Compute embeddings in batches
    start_time_function = time.time()  # Start time for the function

    embeddings = []
    for i in range(0, len(df), BATCH_SIZE):
        start_time = time.time()  # Start time for the batch
        batch = df.iloc[i:i + BATCH_SIZE]

        if isinstance(batch, pd.Series):
            batch = batch.tolist()  # Convert pandas Series to list, if necessary
        elif isinstance(batch, pd.DataFrame):
            batch = batch.squeeze().tolist()  # Convert pandas DataFrame to list, if necessary

        batch = [str(item).lower() for item in batch if item is not None]

        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)

        end_time = time.time()  # End time for the batch
        print(f"Batch {i // BATCH_SIZE + 1} processed in {end_time - start_time:.2f} seconds.")

    embeddings = torch.cat(embeddings, dim=0)

    end_time_function = time.time()  # End time for the function
    print(f"Function processed in {end_time_function - start_time_function:.2f} seconds.")

    return embeddings


def compute_similarity(embedding1, embedding2):
    # Compute similarity matrix in a memory-efficient way
    similarity_scores = cosine_similarity(embedding1,embedding2)
    return similarity_scores


def save_similarity_csv(similarity_scores, filename):
    # Save the similarity scores with market cap categories to CSV
    similarity_df = pd.DataFrame(similarity_scores)
    similarity_df.to_csv(filename)


# Build similarity matrix for STOCKS using sector + industry + sub-industry + description
df_stock = preprocess_and_concatenate(df_stock)  # Preprocess stock data
embed_combined = batch_compute_embeddings(df_stock['combined_text'])  # Compute embeddings
scoring_combined = compute_similarity(embed_combined,embed_combined)  # Compute similarity scoring
save_similarity_csv(scoring_combined,'similarity_combined.csv')  # Save similarity scoring in csv


# Build similarity matrix for STOCKS using sector + industry + sub-industry (simple version)
df_stock = preprocess_and_concatenate(df_stock)  # Preprocess stock data
embed_combined_simple = batch_compute_embeddings(df_stock['combined_text_simple'])  # Compute embeddings
scoring_combined_simple = compute_similarity(embed_combined_simple,embed_combined_simple)  # Compute similarity scoring
save_similarity_csv(scoring_combined_simple,'similarity_combined_simple.csv')  # Save similarity scoring in csv


# Build similarity matrix for SECTOR using sector + industry + sub-industry + description
df_stock = preprocess_and_concatenate(df_stock)  # Preprocess stock data
df_sector = preprocess_and_concatenate(df_sector, type='sector')  # Preprocess sector data
embed_combined = batch_compute_embeddings(df_stock['combined_text'])  # Compute embeddings
embed_sector = batch_compute_embeddings(df_sector['combined_text'])  # Compute embeddings
scoring_sector = compute_similarity(embed_combined,embed_sector)  # Compute similarity scoring
save_similarity_csv(scoring_sector,'similarity_sector.csv')  # Save similarity scoring in csv

# Build similarity matrix for SECTOR using sector + industry + sub-industry (simple version)
df_stock = preprocess_and_concatenate(df_stock)  # Preprocess stock data
df_sector = preprocess_and_concatenate(df_sector, type='sector')  # Preprocess sector data
embed_combined_simple = batch_compute_embeddings(df_stock['combined_text_simple'])  # Compute embeddings
embed_sector = batch_compute_embeddings(df_sector['combined_text'])   # Compute embeddings
scoring_sector_simple = compute_similarity(embed_combined_simple,embed_sector)  # Compute similarity scoring
save_similarity_csv(scoring_sector,'similarity_sector_simple.csv')  # Save similarity scoring in csv



















