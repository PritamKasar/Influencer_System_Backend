from flask import Blueprint, request, jsonify
from app.utils.data_processing import load_and_preprocess_data
from app.utils.recommendations import recommend_top_influencers_graphsage
from app.models.gnn_models import GraphSAGEModel
import torch

influencer_bp = Blueprint('influencer', __name__)

# Load and preprocess data
df, data = load_and_preprocess_data("data/YouTube_real_Dataset_Shuffled.csv")

# Load the trained model
sage_model = GraphSAGEModel(input_dim=4, hidden_dim=16, output_dim=8)
sage_model.load_state_dict(torch.load("data/gnn_sage_model_fixed.pth"))
sage_model.eval()

@influencer_bp.route('/recommend', methods=['GET'])
def recommend_influencers():
    category = request.args.get('category')
    top_n = int(request.args.get('top_n', 5))

    if not category:
        return jsonify({"error": "Category parameter is required"}), 400

    recommendations = recommend_top_influencers_graphsage(category, sage_model, data, df, top_n)
    return jsonify(recommendations.to_dict(orient='records'))
