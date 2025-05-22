import torch
from torchvision import models, transforms
from PIL import Image

from .agent2 import pred

from flask import Flask, request, render_template


device = torch.device("cuda")


def load_model():
    with open("classes.txt") as f:
        classes = f.read().splitlines()

    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("plant_disease_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, classes


def predict():
    model, classes = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open('static/1.jpg').convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        return classes[pred_idx]


app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        file.save(f'static/1.jpg')
        dis = predict().split('___')[1].lower()
        if dis[-1] == '_':
            dis = dis[:-1]
        moist = int(request.form['moist'])
        prediction = pred(moist, dis)
        return render_template('index.html', pred=prediction, dis=dis)
    return render_template('index.html')
