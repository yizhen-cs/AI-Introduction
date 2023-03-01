import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']

# Load the CIFAR10 dataset
(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

x_train = train_x.reshape(50000,1024,3)/255.0
x_test = test_x.reshape(10000,1024,3)/255.0
y_train, y_test = tf.cast(train_y, tf.int16), tf.cast(test_y, tf.int16)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# Build the model
model = keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, input_shape=(1024, 3), return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['sparse_categorical_accuracy'])


# 训练模型
# 批量训练大小为32，迭代5次，测试集比例0.2（48000条训练集数据，12000条测试集数据）
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练前时刻：' + str(nowtime))

history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练后时刻：' + str(nowtime))

# 评估模型
model.evaluate(x_test, y_test, verbose=2)  # 每次迭代输出一条记录，来评价该模型是否有比较好的泛化能力

# 保存整个模型
model.save('CIFAR10_RNN_weights.h5')

# 结果可视化
print(history.history)
loss = history.history['loss']  # 训练集损失
val_loss = history.history['val_loss']  # 测试集损失
acc = history.history['sparse_categorical_accuracy']  # 训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy']  # 测试集准确率

plt.figure(figsize=(10, 3))

plt.subplot(121)
plt.plot(loss, color='b', label='train')
plt.plot(val_loss, color='r', label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc, color='b', label='train')
plt.plot(val_acc, color='r', label='test')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1, 10000)

    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(test_x[num], cmap='gray')
    demo = tf.reshape(x_test[num], (1, 1024, 3))
    y_pred = np.argmax(model.predict(demo))
    plt.title('label:' + str(test_y[num]) + '\npred:' + str(y_pred))

plt.show()