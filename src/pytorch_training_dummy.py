from Algorithmia import ADK
import torch as th
from PIL import Image
from torchvision import transforms
import numpy as np


def process_image(image_url, client):
    local_image = client.file(image_url).getFile().name
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.CenterCrop(32),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(local_image)
    img.load()
    # If the image isn't a square, make it a square
    if img.size[0] != img.size[1]:
        sqrWidth = np.ceil(np.sqrt(img.size[0] * img.size[1])).astype(int)
        image_data = transform(img.resize((sqrWidth, sqrWidth)))
    else:
        image_data = transform(img).unsqueeze()
    image_data = image_data.unsqueeze(dim=0)
    return image_data


def load(state):
    composite = th.load(state.get_model('cifar10'))
    model, classes = composite
    state['model'] = model.cuda()
    state['classes'] = classes
    return state


def apply(input, state):
    """
    Calculates the dot product of two matricies using pytorch, with a cudnn backend.
    Returns the product as the output.
    """
    image_data = process_image(input, state.client)
    image_data = image_data.cuda()
    preds = state['model'](image_data)
    _, predicted = th.max(preds.data, 1)
    predicted = predicted.cpu()
    output = []
    for j in range(len(predicted)):
        prediction = {"class": state['classes'][predicted[j]]}
        output.append(prediction)
    return output


algo = ADK(apply, load)
algo.init("data://algorithmia_admin/DeepFashion_1/willow_example.jpeg")