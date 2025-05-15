import torch
import torch.optim as optim
from app.models.gnn_models import GCNModel, GATModel, GraphSAGEModel
from app.utils.data_processing import load_and_preprocess_data

def train_gnn(model, data, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)  # 8D influencer embeddings
        loss = torch.norm(out, p=2)  # Encourages diverse embeddings
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return loss.item()

def train_and_save_models():
    df, data = load_and_preprocess_data("data/YouTube_real_Dataset_Shuffled.csv")

    # Train GCN Model
    gcn_model = GCNModel(input_dim=4, hidden_dim=16, output_dim=8)
    print("\nTraining GCN...")
    gcn_loss = train_gnn(gcn_model, data)
    torch.save(gcn_model.state_dict(), "data/gnn_gcn_model_fixed.pth")
    print("GCN model training complete and saved!")

    # Train GAT Model
    gat_model = GATModel(input_dim=4, hidden_dim=16, output_dim=8)
    print("\nTraining GAT...")
    gat_loss = train_gnn(gat_model, data)
    torch.save(gat_model.state_dict(), "data/gnn_gat_model.pth")
    print("GAT model training complete and saved!")

    # Train GraphSAGE Model
    sage_model = GraphSAGEModel(input_dim=4, hidden_dim=16, output_dim=8)
    print("\nTraining GraphSAGE...")
    sage_loss = train_gnn(sage_model, data)
    torch.save(sage_model.state_dict(), "data/gnn_sage_model_fixed.pth")
    print("GraphSAGE model training complete and saved!")

if __name__ == "__main__":
    train_and_save_models()
