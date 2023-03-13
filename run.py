from  keras.models import load_model
from inception_restnet import InceptionResNetV1
import mtcnn
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from imageio.v2 import imread
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
image_size = 160
cascade_path= "/Users/datle/Desktop/face_recognition/haarcascade_frontalface_default.xml"
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(cascade_path)

    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)


def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    print(aligned_images.shape)
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def single_pic(model,img):
    aligned_images = np.expand_dims(prewhiten(img), axis=0)
    pd=model.predict_on_batch(aligned_images)
    embs = l2_normalize(np.concatenate(pd))
    return embs

def calc_dist(model,data, new_img):
    x1=[]
    img_resize=cv2.resize(new_img, (image_size,image_size))
    new_img_decode=single_pic(model, img_resize)
    for x in data.keys():
        x1.append(distance.euclidean(data[x], new_img_decode))
    idx=np.array(x1).argmin()
    name = list(data)[idx]
    return name, x1[idx]

def calc_dist_plot(img_name0, img_name1):
    print(calc_dist(img_name0, img_name1))
    plt.subplot(1, 2, 1)
    plt.imshow(imread(data[img_name0]['image_filepath']))
    plt.subplot(1, 2, 2)
    plt.imshow(imread(data[img_name1]['image_filepath']))


# data = {}
# for name in names:
#     image_dirpath = image_dir_basepath + name
#     image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
#     embs = calc_embs(image_filepaths)
#     for i in range(len(image_filepaths)):
#         data['{}{}'.format(name, i)] = {'image_filepath': image_filepaths[i],
#                                         'emb': embs[i]}



model = InceptionResNetV1(
        input_shape=(None, None, 3),
        classes=128,
    )
model.load_weights('facenet_keras_weights.h5')

import glob
import pickle
data={}
img_path= glob.glob("database_img/*.jpeg")
embs = calc_embs(img_path)
for i in range(len(img_path)):
    name = img_path[i].split('/')[-1].split('.')[0]
    data[name]=embs[i]
pickle.dump(data,open('data_face.pkl','wb'))
#
# data= pickle.load(open('data_face.pkl', 'rb'))
# img= cv2.resize(cv2.imread("/Users/datle/Desktop/face_recognition/elon.jpeg"), (image_size, image_size))
# calc_dist(data, img)