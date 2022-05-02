import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)
import keras
print(keras.__version__)
import keras
from keras.applications import vgg16
from keras.preprocessing import image, text, sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, Sequential
from keras.layers import Input, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import cv2

import pickle
# print("---------------------------------------------")
print("Done Importing")
print("---------------------------------------------")
results_df = pd.read_csv('30kreedit.csv')
print("---------------------------------------------")
print("Done Dataset loading")
print(results_df.head())
print("---------------------------------------------")

image_names = results_df['image_name'][::5].values



vgg = vgg16.VGG16(weights='imagenet', include_top=True)
encoder = Model(vgg.input, vgg.layers[-2].output)
print("---------------------------------------------")
print("Done Importing Encoder")
print("---------------------------------------------")

with open('image_features_30k.pickle', 'rb') as handle:
    image_vectors=pickle.load(handle)

print("---------------------------------------------")
print("Done Features Loading")
print("---------------------------------------------")

#process all of the text

VOCAB_SIZE = 10000

tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)

sequenced_comments = ['ssss ' + str(t) + ' eeee' for t in results_df[' comment']]  # add start and end markers to the sentences
tokenizer.fit_on_texts(sequenced_comments)
sequenced_comments = tokenizer.texts_to_sequences(sequenced_comments)
sequenced_comments = np.array(sequenced_comments)

# reshape into an array of the same length of images but with 5 comments per image. 
sequenced_comments = sequenced_comments.reshape(-1,5)

print("---------------------------------------------")
print("Done Text Processing")
print("---------------------------------------------")


# root_path1 = 'sample.jpeg'

print("---------------------------------------------")
print("Done Test image loaded")
print("---------------------------------------------")
decoder = keras.models.load_model("Tam_caption_gen_B64_E200.h5")

print("---------------------------------------------")
print("Done Importing Decoder")
print("---------------------------------------------")
def generate_caption(image_vector):
    cap=''
    """
    Generate an english sentence given an image_vector
    """
    word = 'ssss'
    token = tokenizer.word_index[word]
    #print("Token"+str(token))
    sentence = [word]
    #print("sentence"+str(sentence))
    sequence = [token]
    #print("Seq"+str(sequence))
    
    while word != 'eeee':
#         print("sequence -->"+str(sequence))
#         print(type(sequence))
#         print(type(image_vector))
#         print(sequence)
#         print(image_vector)
#         image_vector=list(image_vector)
#         x=[sequence],[image_vector]
        pred = decoder.predict([np.array([sequence]), np.array([image_vector])]).reshape(-1,VOCAB_SIZE)[-1]
        #pred = decoder.predict([np.array(sequence), np.array(image_vector)]).reshape(-1,VOCAB_SIZE)[-1]
        #pred = decoder1.predict(x).reshape(-1,VOCAB_SIZE)[-1]
        token = np.argmax(pred)
        word = tokenizer.index_word[token]
        sentence.append(word)
        sequence.append(token)
        
    print('generated: ', ' '.join(sentence[1:-1]))
    cap=' '.join(sentence[1:-1])
    # f = open("sample.txt", "w+",encoding='utf-8')
    # f.write(' '.join(sentence[1:-1]))
    # f.close()
    return cap
def load_process_image1(root_path1):
    """
    load and process an image ready to be fed into the pre_build vgg16 encoder.
    """
    print(root_path1)
    img = image.load_img(root_path1)
    img = image.img_to_array(img)
    img = cv2.resize(img, (224,224))
    img = vgg16.preprocess_input(img)
    return img

def vectorize_images1(root_path1):
    image_vectors1 = []
    img = load_process_image1(root_path1)
    image_vectors1.append(encoder.predict(np.expand_dims(img, axis=0)))

    image_vectors1 = np.array(image_vectors1)
    image_vectors1 = image_vectors1.squeeze()    
    return image_vectors1
    

def caption(root_path1):
    image_vectors1 = vectorize_images1(root_path1)
    return  generate_caption(image_vectors1)
#print(image_vectors1)

# img = plt.imread(root_path1)
# plt.imshow(img)
# plt.show()