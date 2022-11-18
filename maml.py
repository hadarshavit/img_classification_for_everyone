import os

import torch
from torchvision.transforms.functional import pil_to_tensor
from  torchvision.transforms.functional_pil import resize
from torch.nn.functional import cross_entropy
from PIL import Image
from learn2learn.algorithms.maml import MAML


class Trainer:
    def __init__(self):
        self.model: MAML = torch.load('model.pt', map_location=torch.device('cpu')).cpu()

    @staticmethod
    def _open_images(path):
        images = []
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            images.append(pil_to_tensor(
                resize(Image.open(full_path), (84, 84))
            ))

        return images

    def fine_tune(self, class1_path, class2_path):
        self.model = self.model.cpu()
        class1_images = self._open_images(class1_path)
        class2_images = self._open_images(class2_path)
        imgs = torch.stack(class1_images + class2_images).float()

        labels = torch.LongTensor([0] * len(class1_images) + [1] * len(class2_images))
        print(imgs.shape, labels.shape)

        loss = cross_entropy(self.model(imgs.cpu()), labels)
        self.model.adapt(loss)

    def inference(self, path):
        img = pil_to_tensor(
            resize(Image.open(path), (84, 84))
            )
        out = self.model(img)
        predictions = out.argmax(dim=1)

        return predictions[0].item()





