from flask import Flask
from flask_cors import CORS
from app.routes.influencer_routes import influencer_bp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins
app.register_blueprint(influencer_bp)

if __name__ == "__main__":
    app.run(debug=True)