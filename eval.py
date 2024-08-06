import torchvision
import torch
from model import TinyVGG
from dataloader import train_data_augmented
from torchvision import transforms
import argparse


parser = argparse.ArgumentParser(description="Classify Xray Image")
parser.add_argument(
    "custom_image_path", type=str, help="Enter The Image you want to classify"
)
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"

custom_image_path = args.custom_image_path
custom_image = torchvision.io.read_image(custom_image_path) / 255
custom_image = custom_image.type(torch.float32)
custom_image = custom_image.unsqueeze(dim=1)

custom_image_transform = transforms.Compose([transforms.Resize(size=(64, 64))])
custom_image_transformed = custom_image_transform(custom_image)


model_1 = TinyVGG(
    input_features=1,
    hidden_features=64,
    output_features=len(train_data_augmented.classes),
).to(device)
model_1 = torch.load
model_1.load_state_dict(torch.load("CUSTOM_XRAY_DETECTION_BEST.pth"))
model_1.eval()
with torch.inference_mode():
    custom_image_pred = model_1(custom_image_transformed.to(device)).to(device)
