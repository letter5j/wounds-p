import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from model import build_whole_model
import torch
import torch.utils.model_zoo as model_zoo
import urllib.request

from torchvision import transforms
import PIL.Image

app = Flask(__name__)

model_p = None

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')
@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)
@app.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory('static/css', path)
@app.route('/static/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory('static/fonts', path)
@app.route('/static/media/<path:path>')
def send_media(path):
    return send_from_directory('static/media', path)
@app.route('/getresult', methods=['GET', 'POST'])
def getresult():
    if request.method == 'POST':
        file = request.files['file']
        # transforms.ToTensor()
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = PIL.Image.open(file).convert('RGB')
        inputs = data_transforms(img)
        inputs.unsqueeze_(0)
        inputs = inputs.to(device)
        
        outputs = model_p(inputs)
        _, preds = torch.max(outputs, 1) 
        print(outputs)
        print(preds)
        className = ['Cellulitis', 'bruises', 'cut', 'mouth', 'sature', 'scrape', 'ulcer']
        i = preds.cpu().numpy().flatten().tolist()[0]
        print(i)
    # return render_template('change_avatar.html')
        return className[i]

if __name__ == "__main__":
    
    if(os.path.exists('model.pth') == False):
        print("model mot found!")
        url = 'https://www.dropbox.com/s/7w5f7iidzmd0029/model.pth?dl=1'
        urllib.request.urlretrieve(url, 'model.pth')

    
    
    print('model is loading!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_whole_model()
    PATH = os.path.abspath(os.path.dirname(__file__))
    model.load_state_dict(torch.load(os.path.join(PATH, 'model.pth'), map_location='cpu'))
    # state_dict = model_zoo.load_url(url)
    # model.load_state_dict(state_dict, map_location='cpu')
    model.to(device)
    model_p = model
    model_p.train(False)
    print('model is loaded!')
    app.run()
