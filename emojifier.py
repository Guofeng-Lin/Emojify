from emo_utils import *
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Input, LSTM, Dropout, Dense, Activation
from tensorflow.python.keras.models import Model

# tensorflow有两种显存分配方式
# 1、仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）
# 2、限制消耗的固定大小的显存（程序不会超出限定的显存大小，若超出就报错）
# 默认是第二种，这里把显存分配方式改成第一种
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 读取GloVe的单词向量表示
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# 读取训练数据集
X_train, Y_train = read_csv('data/train_emoji.csv')
Y_oh_train = convert_to_one_hot(Y_train, C=5)
maxLen = len(max(X_train, key=len).split())


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Create a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    :param word_to_vec_map: dictionary mapping words to their GloVe vector representation.
    :param word_to_index: dictionary mappping words to their indices in the vocabulary.
    :return: embedding_layer: pretrained layer Keras instance
    """

    vocab_size = len(word_to_index) + 1  # adding 1 to fit Keras embedding
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]  # define dimensionality of GloVe word vectors (=50)

    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros
    embedding_matrix = np.zeros((vocab_size, emb_dim))

    # Step 2
    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, index in word_to_index.items():
        embedding_matrix[index, :] = word_to_vec_map[word]

    # Step 3
    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable
    embedding_layer = Embedding(vocab_size, emb_dim)

    # Step 4
    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None, ))

    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer


def emojifier(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the emojifier model's graph

    :param input_shape: shape of the input, usually (max_len, )
    :param word_to_vec_map: dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    :param word_to_index: dictionary mapping from words to their indices in the vocabulary
    :return: a model instance in Keras
    """

    # Define sentence_indices as the input of the graph.
    sentence_indices = Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indices through your embedding layer
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through LSTM layer with 128-dimensional hidden state
    X = LSTM(units=128, recurrent_activation='sigmoid', return_sequences=True)(embeddings)

    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)

    # Propagate X through another LSTM layer with 128-dimensional hidden state
    X = LSTM(units=128, recurrent_activation='sigmoid', return_sequences=False)(X)

    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)

    # Propagate X through a Dense layer with 5 units
    X = Dense(units=5)(X)

    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)

    return model

# 编译构造好的模型
model = emojifier((maxLen, ), word_to_vec_map, word_to_index)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 用数据集训练模型
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C=5)
model.fit(X_train_indices, Y_train_oh, epochs=50, shuffle=True)

# 将模型以h5的格式保存到文件夹中
model.save('models/emojifier.h5')

