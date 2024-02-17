# -*- coding: utf-8 -*-
from models.blip_vqa import blip_vqa
import sys
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from mytranslate import ruen, enru

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_demo_image(img_name, image_size, device):
    raw_image = Image.open(img_name).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


image_size = 480


model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'

model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)


def qa_img(my_image, my_question):
    image = load_demo_image(my_image, image_size=image_size, device=device)
    question = ruen(my_question + '?')
    with torch.no_grad():
        answer = model(image, question, train=False, inference='generate')
        return enru(str(answer[0]))


print(qa_img('test.jpg', 'что на картинке'))
print(qa_img('test.jpg', 'сколько детей в комнате'))
print(qa_img('test.jpg', 'женщина сидит слева или справа'))
print(qa_img('test.jpg', 'что стоит на столе'))
