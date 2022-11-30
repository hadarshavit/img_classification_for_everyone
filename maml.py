import os

import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional_pil import resize
from torch.nn.functional import cross_entropy
from PIL import Image
from learn2learn.algorithms.maml import MAML


class Trainer:
    def __init__(self):
        pass

    def get_model(self, num_classes):
        if num_classes == 2:
            self.model: MAML = torch.load('model.pt', map_location=torch.device('cpu'))
        elif num_classes == 3:
            self.model: MAML = torch.load('maml59999_3.pt', map_location=torch.device('cpu'))
        elif num_classes == 4:
            self.model: MAML = torch.load('maml59999_4.pt', map_location=torch.device('cpu'))
        elif num_classes == 5:
            self.model: MAML = torch.load('maml59999_5.pt', map_location=torch.device('cpu'))

    @staticmethod
    def _open_images(path):
        images = []
        for file in os.listdir(path):
            full_path = os.path.join(path, file)
            images.append(pil_to_tensor(
                resize(Image.open(full_path), [84, 84])
            ))

        return images

    def fine_tune(self, class1_path, class2_path, class3_path, class4_path, class5_path):
        class1_images = self._open_images(class1_path)
        class2_images = self._open_images(class2_path)
        class3_images = self._open_images(class3_path)
        class4_images = self._open_images(class4_path)
        class5_images = self._open_images(class5_path)
        c = 0
        if len(class1_images) > 0:
            c += 1
        if len(class2_images) > 0:
            c += 1
        if len(class3_images) > 0:
            c += 1
        if len(class4_images) > 0:
            c += 1
        if len(class5_images) > 0:
            c += 1

        self.get_model(c)

        imgs = torch.stack(class1_images + class2_images + class3_images + class4_images + class5_images).float()

        labels = torch.LongTensor([0] * len(class1_images) + [1] * len(class2_images) + [2] * len(class3_images) + \
                                  [3] * len(class4_images) + [4] * len(class5_images))
        print(imgs.shape, labels.shape)

        loss = cross_entropy(self.model(imgs), labels)
        self.model.adapt(loss)

    def inference(self, path):
        img = pil_to_tensor(
            resize(Image.open(path), (84, 84))
        ).unsqueeze(0).float()
        out = self.model(img)
        predictions = out.argmax(dim=1)

        return predictions[0].item() + 1
