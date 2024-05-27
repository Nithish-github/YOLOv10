import os
import requests
from ultralytics import YOLOv10

def download_yolov10_weights(model):
    urls = {
        "yolov10n": "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",
        "yolov10s": "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt",
        "yolov10m": "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt",
        "yolov10b": "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt",
        "yolov10x": "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt",
        "yolov10l": "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt"
    }

    # Check if the model is valid
    if model not in urls:
        print(f"Error: model '{model}' is not valid. Choose from: {list(urls.keys())}")
        return
    
    url = urls[model]
    
    # Create the weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    
    file_path = os.path.join('weights', f"{model}.pt")

    # Download the file
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to the file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download: {url}")




#run
#choose the model
model = "yolov10s"
download_yolov10_weights('yolov10s')
model = YOLOv10(f'weights/{model}.pt')
results = model(source=f'images/kobe.jpeg', conf=0.25,save = True)
print(results[0].boxes.xyxy)