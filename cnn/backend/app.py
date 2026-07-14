from flask import Flask, render_template
from flask_cors import CORS
from models.pytorch_model import load_pytorch_model
from services.monitoring import start_monitoring
from routes.main_routes import main_bp
from routes.config_routes import config_bp
from routes.manual_recognize import manual_recognize_bp

app = Flask(__name__)
CORS(app)

load_pytorch_model()
start_monitoring()

app.register_blueprint(main_bp)
app.register_blueprint(config_bp)
app.register_blueprint(manual_recognize_bp)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)