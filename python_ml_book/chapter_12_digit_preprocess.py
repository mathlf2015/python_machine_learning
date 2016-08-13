from chapter_12_loaddata import load_mnist
#导入数据
X_train, y_train = load_mnist('E:/mydata/shuzi_shibie/', kind='train')
X_test,y_test = load_mnist('E:/mydata/shuzi_shibie/', kind='t10k')
print(X_train.shape,X_test.shape)


#数字的可视化
import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train==i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#指定数字的可视化
import matplotlib.pyplot as plt
fig,ax = plt.subplots(5,5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_train[y_train==7][i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()


#将数据存为csv格式
# np.savetxt('train_img.csv', X_train, fmt='%i', delimiter=',')
# np.savetxt('train_labels.csv', y_train, fmt='%i', delimiter=',')
# X_train = np.genfromtxt('train_img.csv', dtype=int, delimiter=',')
# y_train = np.genfromtxt('train_labels.csv', dtype=int, delimiter=',')

# np.savetxt('test_img.csv', X_test, fmt='%i', delimiter=',')
# np.savetxt('test_labels.csv', y_test, fmt='%i', delimiter=',')
# X_test = np.genfromtxt('test_img.csv', dtype=int, delimiter=',')
# y_test = np.genfromtxt('test_labels.csv', dtype=int, delimiter=',')

