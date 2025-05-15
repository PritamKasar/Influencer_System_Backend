import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_top_influencers_graphsage(category, model, data, df, top_n=5):
    category = category.strip().lower()
    filtered_df = df[df['Category'] == category]
    if len(filtered_df) == 0:
        return pd.DataFrame()

    filtered_df = filtered_df.drop_duplicates(subset=['YouTuber Name'])
    influencer_indices = filtered_df.index.values
    influencer_indices = np.intersect1d(influencer_indices, range(len(data.x)))

    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).numpy()

    influencer_embeddings = embeddings[influencer_indices]
    similarity_matrix = cosine_similarity(influencer_embeddings)
    similarity_scores = similarity_matrix.sum(axis=1)

    filtered_df = filtered_df.loc[influencer_indices]
    filtered_df['Similarity Score'] = similarity_scores
    final_recommendations = filtered_df.sort_values(by='Similarity Score', ascending=False).head(top_n).reset_index(drop=True)

    return final_recommendations[['YouTuber Name', 'Subscribers', 'Avg Likes', 'Avg Views', 'Category', 'Similarity Score']]
