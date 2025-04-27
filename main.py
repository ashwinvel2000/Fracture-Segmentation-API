import os
import io
import base64
import pickle
from PIL import Image
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as T
from models import UNet, DoubleConv

app = Flask(__name__)

# Load the model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model_clahe.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(model_path,"rb") as f:
    model = pickle.load(f)
model.to(device)
model.eval()

#Pre- and post-processing
preprocess = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]),
])

def encode_mask(mask_tensor):
    buf = io.BytesIO()
    img = Image.fromarray(mask_tensor.squeeze().cpu().numpy(), mode="L")
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    # 4a) Read image
    img = Image.open(request.files["file"].stream).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(device)  # [1,3,256,256]

    # 4b) Inference
    with torch.no_grad():
        logits = model(x)                       # [1,1,256,256]
        probs  = torch.sigmoid(logits).cpu()
        mask   = (probs > 0.5).type(torch.uint8)*255

    # 4c) Encode and return
    mask_b64 = encode_mask(mask)
    return jsonify({"mask": mask_b64})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)