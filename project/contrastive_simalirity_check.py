import os
import cv2
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

from resnet_model import SupConResNet
from scipy.spatial.distance import cdist


class FeatureModel:
    def __init__(self,
                 img_dir,
                 load_model,
                 model_name='resnet50',
                 weight_path=
                 './save/SupCon/path_models/SupCon_path_resnet50_lr_0.05_decay_0.0001_bsz_8_temp_0.07_trial_0/last.pth',
                 input_size=128,
                 batch_size=8,
                 ):
        self.model_name = model_name
        self.weight_path = weight_path
        self.input_size = input_size
        self.bs = batch_size
        self.img_dir = img_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.normalize = transforms.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.))
        self.transform = transforms.Compose([
            # transforms.PILToTensor(),
            transforms.Resize(size=(input_size, input_size)),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.paths, self.features = self.get_all_features(img_dir)
#         if load_model:
#             self.paths, self.features = self.load_npy(img_dir)
#         else:
#             self.write_npy(img_dir)

    def write_npy(self, img_dir):
        self.paths, self.features = self.get_all_features(img_dir)
        for path, feature in zip(self.paths, self.features):
            name = '.'.join(path.split('.')[:-1]) + self.model_name + ".npy"
            np.save(name, feature)

    def load_model(self, ):
        model = SupConResNet(name=self.model_name)
        model.load_state_dict(torch.load(self.weight_path)['model'])
        model.to(self.device)
        model = model.eval()
        return model

    
    def preprocess_func(self, img_path):
        img = Image.open(img_path)
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to('cpu') #(self.device)
    
    def preprocess_func_2(self, img_path):
        img = Image.open(img_path)
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to('cuda')
    
    # def preprocess_func_2(self, img_path):
    #     img = Image.open(img_path)
    #     tensor = self.transform(img).unsqueeze(0)
    #     return try:
    #         tensor.to(self.device)
    #            except

    def get_feature(self, img_path):
        img = self.preprocess_func_2(img_path)
        with torch.no_grad():
            res = self.model(img)[0].detach().cpu().numpy()
        return res

    def get_features(self, images):
        if type(images) is not torch.Tensor:
            images = np.concatenate(images)
#             print(images.info)
            images = torch.Tensor(images).to(self.device)
        with torch.no_grad():
            res = self.model(images).detach().cpu().numpy()    
        return res

    def load_npy(self, img_dir):
        print('extracting features')
        features = []
        paths = []
        e = 0
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if not file.endswith('jpg') and not file.endswith('png') and not file.endswith('jpeg'):
                    print(file)
                    continue
                img_path = os.path.join(root, file)
                name = '.'.join(img_path.split('.')[:-1]) + ".npy" # + self.model_name + ".npy"
                img = np.load(name)
                features.append(img)
                paths.append(img_path)
                print(f'{e}-{img_path} is done!')
                e += 1
        print('loading features is done!')
        return paths, features

    def get_all_features(self, img_dir):
        print('extracting features')
        features = []
        paths = []
        bs = []
        bs_path = []
        e = 0
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if not file.endswith('jpg') and not file.endswith('png') and not file.endswith('jpeg'):
                    print(file + " is passed!")
                    continue
                img_path = os.path.join(root, file)
                #if Image.open(img_path).mode != 'RGB':
                #       continue
                img = self.preprocess_func(img_path)
                if len(bs) < self.bs:
                    bs.append(img)
                    bs_path.append(img_path)
                else:
                    bs_features = self.get_features(bs)
                    features.extend(bs_features)
                    paths.extend(bs_path)
                    bs = [img]
                    bs_path = [img_path]

#                 print(f'{e}-{img_path} is done!')
                e += 1
        bs_features = self.get_features(bs)
        features.extend(bs_features)
        paths.extend(bs_path)
        print('extracting ' + str(e) + ' features is done!')
        return paths, features

    def get_most_similar(self, img_path, n=10, distance='euclidean'):
        feature = self.get_feature(img_path)
        p = cdist(np.array(self.features),
                  np.expand_dims(feature, axis=0),
                  metric=distance)[:, 0]
        # p = np.sqrt(np.sum((np.array(self.features - feature)) ** 2, axis=1))
        group = zip(p, self.paths.copy())
        res = sorted(group, key=lambda x: x[0])
        r = res[:n]
        return r

    @staticmethod
    def resize_pad(im, desired_size=224):
        old_size = im.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im


if __name__ == '__main__':
    feature_model = FeatureModel(img_dir='cows', load_model=False)
