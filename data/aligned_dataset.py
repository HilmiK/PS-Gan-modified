import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import json


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, 'images', opt.phase)
        self.dir_bbox = os.path.join(opt.dataroot, 'bbox', opt.phase)

        #self.AB_paths, self.bbox_paths = sorted(make_dataset(self.dir_AB, self.dir_bbox))
        self.AB_paths, self.bbox_paths = make_dataset(self.dir_AB, self.dir_bbox)
        self.AB_paths = sorted(self.AB_paths)
        self.bbox_paths = sorted(self.bbox_paths)

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        #print(AB_path)
        bbox_path = self.bbox_paths[index]
        #print(bbox_path)

        w_total = 128 * 2
        w = int(w_total / 2)
        h = 128
        w_offset = random.randint(0, max(0, w - 128 - 1))
        h_offset = random.randint(0, max(0, h - 128 - 1))

        bbox = json.load(open(bbox_path))
        #bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        #print(bbox['y'], bbox['x'], bbox['w'], bbox['h'])
        bbox_x = max(int((bbox['x']/128)*128), 0)
        bbox_y = max(int((bbox['y']/128)*128), 0)
        bbox_w = max(int((bbox['w']/128)*128), 0)
        bbox_h = max(int((bbox['h']/128)*128), 0)

        if bbox_y <= h_offset or bbox_x <= w_offset:
            AB = Image.open(AB_path).convert('RGB')
            AB = AB.resize((256, 128), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, :128,
               :128]
            B = AB[:, :128,
                128:2*128]
            bbox = [bbox['y'], bbox['x'], bbox['w'], bbox['h']]
        else:
            AB = Image.open(AB_path).convert('RGB')
            AB = AB.resize((128* 2, 128), Image.BICUBIC)
            AB = self.transform(AB)
            A = AB[:, h_offset:h_offset + 128,
               w_offset:w_offset + 128]
            B = AB[:, h_offset:h_offset + 128,
                w + w_offset:w + w_offset + 128]
            bbox = [bbox_y-h_offset, bbox_x-w_offset, bbox_w, bbox_h]
        # print('haha')
        # print(bbox)
        

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            #print A.size(2)
            bbox = [bbox[0], A.size(2) - bbox[2], A.size(2) - bbox[1], bbox[3]]
        # print('hehe')
        # print(bbox)
        #print(A.size())
        return {'A': A, 'B': B, 'bbox': bbox,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
