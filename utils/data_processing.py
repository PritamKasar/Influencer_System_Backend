import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import networkx as nx
import torch
from torch_geometric.data import Data

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Category'] = df['Category'].str.lower()
    df = df.dropna(subset=['Audience Country'])

    scaler = MinMaxScaler()
    df[['Subscribers', 'Avg Views', 'Avg Likes', 'Avg Comments']] = scaler.fit_transform(
        df[['Subscribers', 'Avg Views', 'Avg Likes', 'Avg Comments']]
    )

    category_encoder = LabelEncoder()
    df['Category Encoded'] = category_encoder.fit_transform(df['Category'])

    country_encoder = LabelEncoder()
    df['Audience Country Encoded'] = country_encoder.fit_transform(df['Audience Country'])

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['YouTuber Name'], category=row['Category Encoded'],
                   subscribers=row['Subscribers'], views=row['Avg Views'],
                   likes=row['Avg Likes'], comments=row['Avg Comments'])
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i != j and row_i['Category Encoded'] == row_j['Category Encoded']:
                G.add_edge(row_i['YouTuber Name'], row_j['YouTuber Name'])

    node_mapping = {name: idx for idx, name in enumerate(G.nodes())}
    edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(df[['Subscribers', 'Avg Views', 'Avg Likes', 'Avg Comments']].values, dtype=torch.float)
    data = Data(x=node_features, edge_index=edge_index)

    return df, data
