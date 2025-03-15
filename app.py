from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the CSV file
labels_df = pd.read_csv('./sample_images.csv', names=['image_id', 'dx'], delimiter=',')

import torch
import torch.nn as nn
from torchvision import models

# Define the custom fc layer matching the saved model
class CustomFC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CustomFC, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)  # Assuming 512 as hidden size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)  # 7 classes for HAM10000
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features  # Should be 2048 for ResNet50
model.fc = CustomFC(num_ftrs, 7)  # Replace fc with the custom layer

# Load the state dictionary
model.load_state_dict(torch.load('./models/resnet50_ham10000_model2.pth', map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

#Classes: {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Label mapping
label_map = {'bkl': 0, 'nv': 1,'df:': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}
inverse_label_map = {0: 'bkl', 1: 'nv', 2: 'df', 3: 'mel', 4: 'vasc', 5: 'bcc', 6: 'akiec'}

def predict_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
        prediction = inverse_label_map[predicted.item()]
    
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded file
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get actual label from CSV
            image_id = filename.split('.')[0]
            actual_label = labels_df[labels_df['image_id'] == image_id]['dx'].iloc[0]
            
            # Make prediction
            prediction = predict_image(file_path)
            
            return render_template('index.html', 
                                prediction=prediction,
                                actual_label=actual_label,
                                filename=filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
