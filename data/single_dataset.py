import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        print opt.dataroot,"opt.dataroot"
        self.dir_A = os.path.join(opt.dataroot)
        self.dir_A = '/home/hilmi/Desktop/tez/Pedestrian-Synthesis-GAN/datasets/images/test','/home/hilmi/Desktop/tez/Pedestrian-Synthesis-GAN/datasets/bbox/test'
        self.A_paths = make_dataset('/home/hilmi/Desktop/tez/Pedestrian-Synthesis-GAN/datasets/images/test','/home/hilmi/Desktop/tez/Pedestrian-Synthesis-GAN/datasets/bbox/test')

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        print index,"index"
        A_path = self.A_paths[index]

        print A_path

        A_img = Image.open('/home/hilmi/Desktop/tez/Pedestrian-Synthesis-GAN/datasets/images/test/1.jpg').convert('RGB')

        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
