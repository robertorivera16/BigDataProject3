import csv
import json
import re
import string
import tensorflow as tf
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb 
wIndex = imdb.get_word_index()

wIndex = {k:(v+3) for k,v in wIndex.items()}
wIndex["<PAD>"] = 0
wIndex["<START>"] = 1
wIndex["<UNK>"] = 2 
wIndex["<UNUSED>"] = 3

rw_index = dict([(value, key) for (key, value) in wIndex.items()])

def decode_review(text):
    return ' '.join([rw_index.get(i, '?') for i in text])

tw_dict = []
with open("fetched_tweets.txt") as f:
    for i in range(66137):
        line = f.readline()
        tweet = json.loads(line)
        full_text = tweet['text']

        full_text = re.sub(r'[.,"!]+', '', full_text)
        full_text = re.sub(r'^RT[\s]+', '', full_text)
        full_text = re.sub(r'https?:\/\/.[\r\n]', '', full_text)
        full_text = re.sub(r'[:]+', '', full_text)
        new_line = ''
        for i in full_text.split():
            if not i.startswith('@') and not i.startswith('#') and i \
                    not in string.punctuation:
                new_line += i + ' '
        full_text = new_line.lower()

        words = full_text.split()
        result = [] 
        for word in words:
            if word in wIndex:
                result.append(wIndex[word])
        tw_dict.append(result)
train_dataset_data =[]
train_dataset_labels = []
with open('cleantextlabels7.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        data_text = rows[0]
        data_text = data_text.split()
        res = []
        for word in data_text:
            if word in wIndex:
                res.append(wIndex[word])
        train_dataset_data.append(res)
        train_dataset_labels.append((int)(rows[1]))
train_data = keras.preprocessing.sequence.pad_sequences(train_dataset_data,
                                                        value=wIndex["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

tw_dict = keras.preprocessing.sequence.pad_sequences(tw_dict,
                                                        value=wIndex["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

vocab_size = len(wIndex)

model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(6, activation=tf.nn.relu))
model.add(keras.layers.Dense(3, activation=tf.nn.softmax))


model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:5000]
partial_x_train = train_data[5000:]

y_val = train_dataset_labels[:5000]
partial_y_train = train_dataset_labels[5000:]

model.fit(np.array(partial_x_train), np.array(partial_y_train), epochs=40, batch_size=512, validation_data=(np.array(x_val), np.array(y_val)),verbose=1)

res = model.predict_classes(tw_dict)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

countzero = 0
countone = 0
counttwo = 0

for i in range(len(res)):
    if res[i] == 0:
        countzero += 1
    elif res[i] == 1:
        countone += 1
    elif res[i] == 2:
        counttwo += 1

print()
print("Zeroes : " + str(countzero))
print("Ones : " + str(countone))
print("Twos : " + str(counttwo))
print()

print(res[:100])


for i in range(len(res)):
    with open("test_results.txt", "a") as results:
        results.write(str(res[i]))
        results.write("\n")