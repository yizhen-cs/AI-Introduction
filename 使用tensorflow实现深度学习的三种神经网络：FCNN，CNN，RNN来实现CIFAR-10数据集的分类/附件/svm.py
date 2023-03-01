from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 加载数据
(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()

# Flatten images to 2D array
x_train = np.reshape(train_x, (len(train_x), -1))
x_test = np.reshape(test_x, (len(test_x), -1))
# Convert labels to numbers
y_train = np.ravel(train_y)
y_test = np.ravel(test_y)
print(x_train.shape)
print(y_train.shape)

# # 训练 SVM 模型
# clf = LinearSVC(verbose=True, loss='hinge',max_iter=50)
# clf.fit(x_train, y_train)
#
# # Save model
# with open('SVMmodel.pkl', 'wb') as f:
#   pickle.dump(clf, f)

# Load model
with open('SVMmodel.pkl', 'rb') as f:
  clf = pickle.load(f)

# Predict x
# y_pred = clf.predict(x)
# 评估模型
accuracy = clf.score(x_train, y_train)
print("Train Accuracy: {:.2f}".format(accuracy))

# Compute predictions
predictions = clf.decision_function(x_train)

# Compute loss
loss = np.maximum(0, 1 - train_y * predictions)

# Print mean loss
print("Train Loss: {:.2f}".format(np.mean(loss)))

# 评估模型
accuracy = clf.score(x_test, y_test)
print("Test Accuracy: {:.2f}".format(accuracy))

# Compute predictions
predictions = clf.decision_function(x_test)

# Compute loss
loss = np.maximum(0, 1 - test_y * predictions)

# Print mean loss
print("Test Loss: {:.2f}".format(np.mean(loss)))


# 使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1, 10000)

    plt.subplot(2, 5, i + 1)
    plt.axis('off')
    plt.imshow(test_x[num], cmap='gray')
    demo = np.reshape(x_test[num], (1, 3072))
    y_pred = clf.predict(demo)
    plt.title('label:' + str(test_y[num]) + '\npred:' + str(y_pred))


# plt.ion()       #打开交互式操作模式#
plt.show()

# plt.pause(5)
# plt.close()



