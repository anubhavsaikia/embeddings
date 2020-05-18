import numpy as np
import tensorflow as tf
import nltk
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

f1 = open('corpus1.txt','r')
contents = f1.read()
texts= nltk.sent_tokenize(contents)
texts = [x for x in texts if len(x.split())>2]
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_to_index = tokenizer.word_index
vocab_size = len(word_to_index)
print(vocab_size)
x_train = []
y_train = []
one_zero_train_ratio =7
for seq in sequences:
    se_det = min(len(seq)-2,5)
    start_index = int((len(seq)-se_det)/2)
    end_index = start_index + se_det -1
    for i in range(start_index, end_index+1):
        left_boundary = max(0,i-4)
        right_boundary = min(i+4,len(seq)-1)
        context= seq[i]
        left_list = seq[left_boundary:i]
        right_list = seq[i+1:right_boundary+1]
        choose_from = left_list+right_list
        rand_index=np.random.randint(len(choose_from),size=(1,))[0]
        target = choose_from[rand_index]
        x_train.append([context,target])
        y_train.append(1)
        iter = 0
        while iter < one_zero_train_ratio:
            ran_num = np.random.randint(1,vocab_size+1,size=(1,))[0]
            if ran_num != target:
                x_train.append([context,ran_num])
                y_train.append(0)
                iter = iter+1
x_train = np.array(x_train, dtype="float64")
y_train = np.array(y_train, dtype="float64")
print(np.shape(x_train))
print(np.shape(y_train))
## TRAINING DATA PREPARED

input_layer = keras.Input(shape = (2,), name = "InputLayer")

embedding_layer_init = layers.Embedding(vocab_size+1, 100)
embedding_layer = embedding_layer_init(input_layer)

lstm_one = layers.LSTM(100, return_sequences = True)(embedding_layer)
lstm_one = layers.Dropout(0.1)(lstm_one)

lstm_two = layers.LSTM(100)(lstm_one)
lstm_two = layers.Dropout(0.1)(lstm_two)

first_dense = layers.Dense(100, name = "firstDense")(lstm_two)
#first_dense = layers.Dropout(0.1)(first_dense)

second_dense = layers.Dense(100, name = "secondDense")(first_dense)
second_dense = layers.Dropout(0.1)(second_dense)

third_dense = layers.Dense(20,name= "thirdDense")(second_dense)
#third_dense = layers.Dropout(0.2)(third_dense)

output_layer = layers.Dense(2,name="outputLayer", activation="softmax")(third_dense)

model = keras.Model(
    inputs=[input_layer],
    outputs=[output_layer],
)

keras.utils.plot_model(model, "embedding_model.png", show_shapes=True)
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)

model.compile(
    optimizer=opt,               
    loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    {"InputLayer": x_train},
    {"outputLayer": y_train},
    epochs=1,
    batch_size=50
)

model.save('embedding_one')

embeddings = embedding_layer_init.get_weights()[0]
words_embeddings = {w:embeddings[idx] for w, idx in word_to_index.items()}

f1 = open('my_embeddings_4.txt','w+')
em_text=""
for w, vec in words_embeddings.items():
    new_vec  = [str(x) for x in vec]
    vec_string =  ' '.join(new_vec)
    
    em_text =em_text+w+" : "+vec_string+'\n'

f1.write(em_text)









