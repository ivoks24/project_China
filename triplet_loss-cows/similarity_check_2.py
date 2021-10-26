import shutil
import os
from os.path import join
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import Normalizer


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


def get_resnet50_triplet(vec_dim, weight_path):
    input_1 = Input(shape=(None, None, 3))
    input_2 = Input(shape=(None, None, 3))
    input_3 = Input(shape=(None, None, 3))

    x1 = tf.keras.applications.resnet.preprocess_input(input_1)
    x2 = tf.keras.applications.resnet.preprocess_input(input_2)
    x3 = tf.keras.applications.resnet.preprocess_input(input_3)
    base_model = tf.keras.applications.resnet.ResNet50(weights=None,
                                                       include_top=False,
                                                       pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x1 = base_model(x1)
    x2 = base_model(x2)
    x3 = base_model(x3)
    layer_normalizer = tf.keras.layers.LayerNormalization(name='layer_normalization')

    x1 = layer_normalizer(x1)
    x2 = layer_normalizer(x2)
    x3 = layer_normalizer(x3)

    dense_1 = Dense(128, activation="linear", name="dense_image_1", use_bias=False)

    x1 = dense_1(x1)
    x2 = dense_1(x2)
    x3 = dense_1(x3)
    _norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    x1 = _norm(x1)
    x2 = _norm(x2)
    x3 = _norm(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    model = Model([input_1, input_2, input_3], x)

    # model.compile(loss=triplet_loss, optimizer=Adam(lr))

    model.summary()
    model.load_weights(weight_path)

    y = tf.keras.applications.resnet.preprocess_input(input_1)
    y = model.get_layer('resnet50')(y)
    y = model.get_layer('layer_normalization')(y)
    y = model.get_layer('dense_image_1')(y)
    y = _norm(y)
    new_model = tf.keras.models.Model(input_1, y)
    return new_model


def get_mobilenetv2_triplet(vec_dim, weight_path):
    input_1 = Input(shape=(None, None, 3))
    input_2 = Input(shape=(None, None, 3))
    input_3 = Input(shape=(None, None, 3))

    x1 = tf.keras.applications.mobilenet_v2.preprocess_input(input_1)
    x2 = tf.keras.applications.mobilenet_v2.preprocess_input(input_2)
    x3 = tf.keras.applications.mobilenet_v2.preprocess_input(input_3)
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None,
                                                                include_top=False,
                                                                pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x1 = base_model(x1)
    x2 = base_model(x2)
    x3 = base_model(x3)
    layer_normalizer = tf.keras.layers.LayerNormalization(name='layer_normalization')

    x1 = layer_normalizer(x1)
    x2 = layer_normalizer(x2)
    x3 = layer_normalizer(x3)

    dense_1 = Dense(64, activation="linear", name="dense_image_1", use_bias=False)

    x1 = dense_1(x1)
    x2 = dense_1(x2)
    x3 = dense_1(x3)
    _norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))

    x1 = _norm(x1)
    x2 = _norm(x2)
    x3 = _norm(x3)
    x = Concatenate(axis=-1)([x1, x2, x3])
    model = Model([input_1, input_2, input_3], x)

    # model.compile(loss=triplet_loss, optimizer=Adam(lr))

    model.summary()
    model.load_weights(weight_path)

    y = tf.keras.applications.resnet.preprocess_input(input_1)
    y = model.get_layer('mobilenetv2_1.00_None')(y)
    y = model.get_layer('layer_normalization')(y)
    y = model.get_layer('dense_image_1')(y)
    y = _norm(y)
    new_model = tf.keras.models.Model(input_1, y)
    return new_model


class FeatureModel:
    def __init__(self,
                 weight_path,
                 img_dir,
                 load_model,
                 input_size,
                 batch_size=128,
                 n_cluster=20,
                 ):
        self.input_size = input_size
        self.bs = batch_size
        self.n_clusters = n_cluster
        self.img_dir = img_dir
        self.model = self.load_model(input_size, weight_path)
        self.l2_normalizer = Normalizer('l2')
        if load_model:
            self.paths, self.features = self.load_npy(img_dir)
        else:
            self.write_npy(img_dir)

    def write_npy(self, img_dir):
        self.paths, self.features = self.get_all_features(img_dir)
        # for path, feature in zip(self.paths, self.features):

    @staticmethod
    def load_model(input_size, weight_path):
        print('loading model')
        model = get_resnet50_triplet(input_size, weight_path)
        return model

    def preprocess_func(self, img_path):
        img = cv2.imread(img_path)[..., ::-1]
        img = self.resize_pad(img, self.input_size[0])
        return img

    def get_feature(self, img_path):
        img = self.preprocess_func(img_path)
        img = np.expand_dims(img, axis=0)
        return self.model.predict(img)[0]

    def get_features(self, images):
        if type(images) is not np.ndarray:
            images = np.array(images)
        return self.model.predict(images)

    def load_npy(self, img_dir):
        print('extracting features')
        features = []
        paths = []
        e = 0
        for cow_name in os.listdir(img_dir):
            cow_dir = join(img_dir, cow_name)
            feature_file = join(cow_dir, 'feature.npy')
            if os.path.isfile(feature_file):
                feature = np.load(feature_file)
                features.append(feature)
                paths.append(cow_dir)
                print(f'{e}-{cow_dir} is done!')
                e += 1
        print('loading features is done!')
        return paths, features

    def get_all_features(self, img_dir):
        print('extracting features is started')
        features = []
        paths = []
        bs = []
        e = 0
        for cow_name in os.listdir(img_dir):
            cow_dir = join(img_dir, cow_name)
            feature_path = join(cow_dir, f"feature.npy")
            if os.path.isfile(feature_path):
                features.append(np.load(feature_path))
                paths.append(feature_path)
                continue
            for file_name in os.listdir(cow_dir):
                file_path = join(cow_dir, file_name)
                if not file_path.endswith('jpg') and not file_path.endswith('png') and not file_path.endswith('jpeg'):
                    print(file_path + " is passed!")
                    continue
                img = self.preprocess_func(file_path)
                bs.append(img)
            if bs:
                bs_features = self.get_features(bs)
                encode = np.sum(bs_features, axis=0)
                encode = self.l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                features.append(encode)
                paths.append(cow_dir)
                np.save(feature_path, encode)
                print(f'{e}-{cow_dir} is done!')
            e += 1
        print('extracting features is done!')
        return paths, features

    def get_most_similar(self, img_path, n=10, distance='euclidean', threshold=None, save_unknown=False):
        feature = self.get_feature(img_path)
        p = cdist(np.array(self.features),
                  np.expand_dims(feature, axis=0),
                  metric=distance)[:, 0]
        group = zip(p, self.paths.copy())
        if threshold is not None:
            group = [(p, g) for p, g in group if p < threshold]
            if len(group) == 0 and save_unknown:
                new_cow = len(os.listdir(self.img_dir))
                new_cow_path = join(self.img_dir, "new_cow_" + str(new_cow))
                new_feature_path = join(new_cow_path, 'feature.npy')
                encode = self.l2_normalizer.transform(np.expand_dims(feature, axis=0))[0]
                os.makedirs(new_cow_path)
                np.save(new_feature_path, encode)
                shutil.copy(img_path, new_cow_path)
                print(f'{new_cow_path} has been created')
                self.features.append(encode)
                self.paths.append(new_cow_path)
        res = sorted(group, key=lambda x: x[0])
        r = res[:n]

        return r

    @staticmethod
    def resize_pad(im, desired_size=224):
        old_size = im.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return new_im


if __name__ == '__main__':
    feature_model = FeatureModel('cowsResnet50.h5',
                                 img_dir='cows',
                                 load_model=False,
                                 input_size=(224, 224, 3))
    feature_model.get_most_similar('cows-test/14/14-1.jpg', threshold=0.05, save_unknown=True)
