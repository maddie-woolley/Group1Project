# code to create heatmap visualizations
# using pretrained resnet 18 model, so similar but not 100% same
# Vaishnav, R. (2021, June 28). Visualizing Feature Maps Using Pytorch. Medium.
# Retrieved November 28, 2022,
# from https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

image = Image.open('10171444_near_ir.jpg')
plt.imshow(image)

model = models.resnet18(pretrained=True)
print(model)

model_weights = []
conv_layers = []
model_children = list(model.children())
counter = 0

list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

image = transform(image)
# print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
# print(f"Image shape after: {image.shape}")
image = image.to(device)

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
print(len(outputs))
#print feature_maps
# for feature_map in outputs:
    # print(feature_map.shape)
processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
# for fm in processed:
   # print(fm.shape)

fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
