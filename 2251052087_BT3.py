from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer, fetch_lfw_people, fetch_20newsgroups
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import mglearn
import ssl


#
# iris = datasets.load_iris()
# X, y = iris.data[:, :2], iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(accuracy_score(y_test, y_pred))

#1
iris = datasets.load_iris()
print(iris.data[:5])

#2
print(iris.target)
for i in range(len(iris.target_names)):
    print(f"{i}:{iris.target_names[i]}")

#3
sepal_length = iris.data[:, 0]
sepal_width = iris.data[:, 1]

plt.scatter(sepal_length[iris.target == 0], sepal_width[iris.target == 0], label='Setosa', c='red')
plt.scatter(sepal_length[iris.target == 1], sepal_width[iris.target == 1], label='Versicolor', c='blue')
plt.scatter(sepal_length[iris.target == 2], sepal_width[iris.target == 2], label='Virginica', c='green')

plt.xlabel('Chiều dài đài hoa(cm)')
plt.ylabel('Chiều rộng đài hoa(cm)')
plt.title('Biểu đồ phân tán của các loài Iris')
plt.legend()
plt.show()

#4
pca = PCA(n_components=3)
iris_pca = pca.fit_transform(iris.data)
print('Dữ liệu sau khi giảm chiều(5 hàng đầu tiên): ')
print(iris_pca[:5])

#5
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=140, test_size=10,
                                                    random_state=42)
print('Kích thước tập huấn luyện: ', X_train.shape)
print('Kích thước tập kiểm tra: ', X_test.shape)

#6
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Dự đoán: ', y_pred)

#7
print('Nhãn thực tế: ', y_test)
print('Nhãn dự đoán: ', y_pred)
accuracy = accuracy_score(y_test, y_pred)
print('Độ chính xác: ', accuracy)

#8
X = iris.data[:, [0, 1]]
y = iris.target
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Chiều dài đài hoa (cm)')
plt.ylabel('Chiều rộng đài hoa (cm)')
plt.title('Ranh giới quyết định với KNN')
plt.show()

#9
diabetes = load_diabetes()
print('Dữ liệu (5 hàng đầu tiên): ')
print(diabetes.data[:5])

#10
X_train = diabetes.data[:422]
X_test = diabetes.data[422:]
y_train = diabetes.target[:422]
y_test = diabetes.target[422:]
print('Kích thước tập huấn luyện: ', X_train.shape)
print('Kích thước tập kiêm tra: ', X_test.shape)

#11
lr = LinearRegression()
lr.fit(X_train, y_train)

#12
coefficients = lr.coef_
print('10 hệ số b: ', coefficients)

#13
y_pred = lr.predict(X_test)
print('Giá trị thực tế: ', y_test)
print('Giá trị dự đoán: ', y_pred)

#14
r2 = r2_score(y_test, y_pred)
print('Hệ số R^2:', r2)
mse = mean_squared_error(y_test, y_pred)
print('Lỗi trung bình bình phương (MSE): ', mse)

#15
X_train_age = X_train[:, [0]]
X_test_age = X_test[:, [0]]
lr_age = LinearRegression()
lr_age.fit(X_train_age, y_train)
y_pred_age = lr_age.predict(X_test_age)
print('Dự đoán với tuổi: ', y_pred_age)

#16
fig, axes = plt.subplots(2, 5, figsize=(20, 10))

axes = axes.flatten()

for i in range(10):
    X_train_single = X_train[:, [i]]
    X_test_single = X_test[:, [i]]

    lr_single = LinearRegression()
    lr_single.fit(X_train_single, y_train)

    y_pred_single = lr_single.predict(X_test_single)

    ax = axes[i]
    ax.scatter(X_test_single, y_test, color='blue', label='Thực tế')
    ax.plot(X_test_single, y_pred_single, color='red', label='Dự đoán')
    ax.set_xlabel(f'Đặc trưng {i + 1}')
    ax.set_ylabel('Tiến trình bệnh')
    ax.set_title(f'Hồi quy tuyến tính với đặc trưng {i + 1}')
    ax.legend()

plt.tight_layout()
plt.show()

#17
breast_cancer = load_breast_cancer()
print('Các khóa của từ điển: ', breast_cancer.keys())

#18
print('Kích thước của dữ liệu: ', breast_cancer.data.shape)
target_series = pd.Series(breast_cancer.target)
benign_count = target_series.value_counts()[1]
malignant_count = target_series.value_counts()[0]

print('Số lượng khối u lành tính (benign):', benign_count)
print('Số lương khối u ác tính (malignant): ', malignant_count)

#19
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2,
                                                    random_state=42)
train_scores = []
test_scores = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# Trực quan hóa
plt.plot(range(1, 11), train_scores, label='Độ chính xác của tập huấn luyện')
plt.plot(range(1, 11), test_scores, label='Độ chính xác tập kiểm tra')
plt.xlabel('Số láng giềng (K)')
plt.ylabel('Độ chính xác')
plt.title('Hiệu suất KNN với số láng giềng từ 1 đến 10')
plt.legend()
plt.show()

#20
X, y = mglearn.datasets.make_forge()

logreg = LogisticRegression().fit(X, y)
print('Độ chính xác Logistic Regression: ', logreg.score(X, y))
svc = LinearSVC().fit(X, y)
print('Độ chính xác Lineear SVC: ', svc.score(X, y))

# ssl._create_default_https_context = ssl._create_unverified_context
#
# #21
# faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# print('Mô tả bộ dữ liệu:\n', faces.DESCR)
#
# #22
# print('Kích thước images: ', faces.images.shape)
# print('Kích thước data: ', faces.data.shape)
# print('Kích thước target: ', faces.target.shape)
# print('Tên nhãn: ', faces.target_names)
#
# #23
# def plot_faces(images, n_row=2, n_col=5):
#     plt.figure(figsize=(2 * n_col, 2.5 * n_row))
#     for i in range(n_row * n_col):
#         plt.subplot(n_row, n_col, i + 1)
#         plt.imshow(images[i], cmap='gray')
#         plt.axis('off')
#     plt.show()
#
#
# # Vẽ 10 khuôn mặt
# plot_faces(faces.images)
#
# #24
# svc = SVC(kernel='linear')
#
# #25
# X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=42)
# print('Kích thước tập huấn luyện: ', X_train.shape)
# print('Kích thước tập kiếm tra: ', X_test.shape)
#
# #26
# def evaluate_cross_validation(model, X, y, k=5):
#     scores = cross_val_score(model, X, y, cv=k)
#     print(f'Độ chính xác K-fold (k={k}:{scores.mean():.2f}(+/-{scores.std() * 2:.2f})')
#
# #27
# def train_and_evaluate(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     train_scores = model.score(X_train, y_train)
#     test_scores = model.score(X_test, y_test)
#     print('Độ chính xác tập huấn luyện: ', train_scores)
#     print('Độ chính xác tập kiểm tra: ', test_scores)
#
# #28
# svc = SVC(kernel='linear')
#
# evaluate_cross_validation(svc, faces.data, faces.target)
# train_and_evaluate(svc, X_train, X_test, y_train, y_test)
#
# #29
# def create_glasses_target(target):
#     np.random.seed(42)
#     glasses_target = np.random.randint(0, 2, size=len(target))
#     return glasses_target
#
# faces_glasses_target = create_glasses_target(faces.target)
# print('Mảng mục tiêu mới (10 giá trị đầu tiên):', faces_glasses_target[:10])
#
# #30
# X_train, X_test, y_train, y_test = train_test_split(faces.data, faces_glasses_target, test_size=0.25, random_state=42)
# svc_2 = SVC(kernel='linear')
# svc_2.fit(X_train, y_train)
#
# #31
# def evaluate_cross_validation(model, X, y, k=5):
#     scores = cross_val_score(model, X, y, cv=k)
#     print(f'Độ chính xác K-fold (k={k}:{scores.mean():.2f}(+/-{scores.std() * 2:.2f})')
#
#
# svc_2 = SVC(kernel='linear')
# evaluate_cross_validation(svc_2, X_train, y_train, 5)
#
# #32
# X_eval = faces.data[30:40]
# y_eval = faces_glasses_target[30:40]
#
# X_train_remaining = np.concatenate((faces.data[:30], faces.data[:40]))
# y_train_remaining = np.concatenate((faces_glasses_target[:30], faces_glasses_target[:40]))
#
# svc_3 = SVC(kernel='linear')
# svc_3.fit(X_train_remaining, y_train_remaining)
#
# accuracy = svc_3.score(X_eval, y_eval)
# print('Độ chính xác trên tập đánh giá 10 ảnh: ', accuracy)
#
# #33
# y_pred = svc_3.predict(X_eval)
#
# eval_faces = [np.reshape(a, (50, 37)) for a in X_eval]
#
#
# def plot_faces(images, predictions, n_col=10):
#     plt.figure(figsize=(2 * n_col, 2.5))
#     for i in range(len(images)):
#         plt.subplot(1, n_col, i + 1)
#         plt.imshow(images[i], cmap='gray')
#         plt.title(f'pred:{predictions[i]}')
#         plt.axis('off')
#     plt.show()
#
#
# plot_faces(eval_faces, y_pred)
# for i in range(len(y_eval)):
#     if y_eval[i] != y_pred[i]:
#         print(f'Ảnh ở chỉ ố {i + 30} bị phân loại sai. Thực tế: {y_eval[i]}, Dự đoán: {y_pred[i]}')
#
# #34
# news = fetch_20newsgroups(subset='all')
# print(f"Số lượng bài báo: {len(news.data)}")
# print(f"Các chủ đề: {news.target_names[:5]}")
#
#
# #35
# print(type(news.data))
# print(type(news.target))
# print(type(news.target_names))
#
# print("Danh sách các chủ đề:")
# print(news.target_names)
#
# #36
# X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=42)
#
# print(f"Số bài báo trong tập huấn luyện: {len(X_train)}")
# print(f"Số bài báo trong tập kiểm tra: {len(X_test)}")