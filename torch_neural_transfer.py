from __future__ import print_function

import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models


num_steps = 300
style_weight = 1000000
content_weight = 1
buffer_size = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = (buffer_size, buffer_size) if torch.cuda.is_available() else (128, 128)

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

warnings.filterwarnings("ignore")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
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


def im_save(tensor, title):
    """
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(title, 'PNG')


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
    return G.div(a * b * c * d)


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
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
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


def get_input_optimizer(input_img):
    """
    """
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps,
                       style_weight, content_weight):
    """
    """
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            """
            """
            with torch.no_grad():
                input_img.clamp_(0, 1)
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
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img


if __name__ == "__main__":
    """
    """
    d_path = {}
    d_path['content'] = tf.keras.utils.get_file('kyoto.jpg',
                            # 'https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg',
                            'https://wallpaperset.com/w/full/6/2/b/124421.jpg',
                            )
    d_path['style'] = tf.keras.utils.get_file('starry_night.jpg',
                            # 'https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg',
                            'https://wallpaperset.com/w/full/2/2/9/207342.jpg',
                            )

    style_img = image_loader(d_path['style'])[:, :, :, :buffer_size]
    content_img = image_loader(d_path['content'])[:, :, :, :buffer_size]
    input_img = content_img.clone()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    output = run_style_transfer(cnn, cnn_normalization_mean, 
                                cnn_normalization_std, content_img, 
                                style_img, input_img, num_steps=num_steps, 
                                style_weight=style_weight, content_weight=content_weight)

    im_save(output, title='./out/neural_image.png')