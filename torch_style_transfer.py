import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tensorflow as tf

from torchviz import make_dot
from PIL import Image
import matplotlib.pyplot as plt
import warnings


buffer_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

imsize = (buffer_size, buffer_size) if torch.cuda.is_available() else (128, 128)

loader = torchvision.transforms.Compose([
        torchvision.transforms.Resize(imsize),
        torchvision.transforms.ToTensor()])

unloader = torchvision.transforms.ToPILImage() 

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def image_loader(image_name):
    """
    """
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow_tensor(tensor, ax=None):
    """
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save('./out/output_image.png', 'PNG')
    if ax:
        ax.imshow(image)
    else:
        plt.imshow(image)


class ContentLoss(nn.Module):
    """
    """
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """
    """
    a, b, c, d = input.size() 
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  
    G_norm = G.div(a * b * c * d)
    return G_norm


class StyleLoss(nn.Module):
    """
    """
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """
    """
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    """
    """
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for n_child, layer in enumerate(cnn.children()):
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


def closure():
    """
    """
    input_img.data.clamp_(0, 1)
    optimizer.zero_grad()
    model(input_img)
    style_score = 0
    content_score = 0
    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss
    style_score *= style_weight
    content_score *= content_weight
    loss = style_score + content_score
    loss.backward()
    run[0] += 1
    if run[0] % 2 == 0:
        print("run {}:".format(run))
        print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            style_score.item(), content_score.item()))
        input_img.data.clamp_(0, 1)
        d_images[run[0]] = input_img
        print()
    return style_score + content_score


def get_input_optimizer(input_img):
    """
    """
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    return optimizer


if __name__ == "__main__":
    """
    """
    d_path = {}
    d_path['content'] = tf.keras.utils.get_file('kyoto.jpg','https://wallpaperset.com/w/full/6/2/b/124421.jpg')
    d_path['style'] = tf.keras.utils.get_file('starry_night.jpg','https://wallpaperset.com/w/full/2/2/9/207342.jpg')
    # d_path['style'] = tf.keras.utils.get_file('red_city.jpg', 'https://wallpaperset.com/w/full/a/9/1/79298.jpg')

    style_img = image_loader(d_path['style'])[:, :, :, :buffer_size]
    content_img = image_loader(d_path['content'])[:, :, :, :buffer_size]
    input_img = content_img.clone()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    warnings.filterwarnings("ignore")
    cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
    model, style_losses, content_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    num_steps = 20
    style_weight=5000
    content_weight=1
    input_img = content_img[:, :, :, :buffer_size].clone()
    d_images = {}

    print('\nBuilding the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        cnn_normalization_mean, cnn_normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:
        optimizer.step(closure)
        input_img.data.clamp_(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    d_img = {"Content": content_img,
            "Style": style_img,
            "Output": input_img}

    for i, key in enumerate(d_img.keys()):
        imshow_tensor(d_img[key], ax=axes[i])
        axes[i].set_title(f"{key} Image")
        axes[i].axis('off')