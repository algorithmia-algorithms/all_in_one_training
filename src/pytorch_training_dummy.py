import Algorithmia
import torch as th
from PIL import Image
from torchvision import transforms
import numpy as np


def process_image(image_url, client):
    local_image = client.file(image_url).getFile().name
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(local_image)
    img.load()
    # If the image isn't a square, make it a square
    if img.size[0] != img.size[1]:
        sqrWidth = np.ceil(np.sqrt(img.size[0] * img.size[1])).astype(int)
        image_data = transform(img.resize((sqrWidth, sqrWidth)))
    else:
        image_data = transform(img)
    return image_data


def load(state):
    state['classes'], state['model'] = th.load(state.get_model('cifar10'))
    return state


def apply(input, state):
    """
    Calculates the dot product of two matricies using pytorch, with a cudnn backend.
    Returns the product as the output.
    """
    image_data = process_image(input, state.client)
    preds = state['model'](image_data)
    _, predicted = th.max(preds.data, 1)
    output = []
    for j in range(len(state['classes'])):
        prediction = {"class": state['classes'][preds[j]]}
        output.append(prediction)
    return output
