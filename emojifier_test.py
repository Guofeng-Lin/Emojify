from emo_utils import *
import tensorflow as tf
from tensorflow.python.keras.models import load_model

# tensorflow有两种显存分配方式
# 1、仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）
# 2、限制消耗的固定大小的显存（程序不会超出限定的显存大小，若超出就报错）
# 默认是第二种，这里把显存分配方式改成第一种
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 读取GloVe的单词向量表示
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# 读取训练数据集和测试数据集
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())

# 将测试数据转换为模型可以接受的格式
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len=maxLen)
Y_oh_test = convert_to_one_hot(Y_test, C=5)

# 读取模型
model = load_model('models/emojifier.h5')

# 测试模型
loss, acc = model.evaluate(X_test_indices, Y_oh_test)
print()
print("Test accuracy = ", acc)
