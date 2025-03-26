from xml.etree.ElementPath import xpath_tokenizer_re
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

ssl._create_default_https_context = ssl._create_unverified_context
# #Chuẩn bị tiền xử lý
# iris = datasets.load_iris()
# X, y = iris.data[:, :2], iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#
# #xây dựng model
# knn = neighbors.KNeighborsClassifier(n_neighbors=5)
#
# #Training
# knn.fit(X_train, y_train)
#
# #Dự đoán
# y_pred = knn.predict(X_test)
#
# #Đánh giá
# accuracy_score(y_test, y_pred)
# print(f"Độ chính xác của mô hình KNN: {accuracy_score(y_test, y_pred)}")

# Bai 1
# Sepal length (Chiều dài đài hoa),Sepal width (Chiều rộng đài hoa),Petal length (Chiều dài cánh hoa),Petal width (Chiều rộng cánh hoa)]
print("Bài 1: ")
iris = datasets.load_iris()
print(iris.data[:5])

print('Bài 2: ')
print(iris.target)
for i in range(len(iris.target_names)):
    print(f"{i}:{iris.target_names[i]}")

print('Bài 3:')
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

print('Bài 4')
# Khởi tạo pca với 3 thành phần
pca = PCA(n_components=3)
# Áp dụng pca lên dữ liệu
iris_pca = pca.fit_transform(iris.data)
print('Dữ liệu sau khi giảm chiều(5 hàng đầu tiên): ')
print(iris_pca[:5])

print('Bài 5')
# Chia dữ liệu thành 140 mẫu huấn luyện và 10 mẫu kiểm tra
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=140, test_size=10,
                                                    random_state=42)
print('Kích thước tập huấn luyện: ', X_train.shape)
print('Kích thước tập kiểm tra: ', X_test.shape)

print('Bài 6')
# Khởi tạo knn với K=5
knn = KNeighborsClassifier(n_neighbors=5)
# Huấn luyện mô hình
knn.fit(X_train, y_train)
# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)
print('Dự đoán: ', y_pred)

print('Bài 7: ')
# So sánh
print('Nhãn thực tế: ', y_test)
print('Nhãn dự đoán: ', y_pred)
# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print('Độ chính xác: ', accuracy)

print('Bài 8: ')
# Chỉ sử dụng 2 đặc trưng đài hoa
X = iris.data[:, [0, 1]]  # Chiều dài đài hoa và chiều rộng đài hoa
y = iris.target
# Tạo lưới tọa độ
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# Huấn luyện KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
# Dự đoán trên lưới
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Vẽ ranh giới
plt.contour(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Chiều dài đài hoa (cm)')
plt.ylabel('Chiều rộng đài hoa (cm)')
plt.title('Ranh giới quyết định với KNN')
plt.show()

print('Bài 9:')
# Tải bộ dữ liệu diabetes
diabetes = load_diabetes()
print('Dữ liệu (5 hàng đầu tiên): ')
print(diabetes.data[:5])

print('Bài 10:')
# Chia dữ liệu
X_train = diabetes.data[:422]
X_test = diabetes.data[422:]
y_train = diabetes.target[:422]
y_test = diabetes.target[422:]
print('Kích thước tập huấn luyện: ', X_train.shape)
print('Kích thước tập kiêm tra: ', X_test.shape)

print('Bài 11: ')
# Khởi tạo mô hình
lr = LinearRegression()
# Huấn luyeện mô hình
lr.fit(X_train, y_train)

print('Bài 12: ')
# Lấy hệ số
coefficients = lr.coef_
print('10 hệ số b: ', coefficients)

print('Bài 13: ')
# Dự đoán
y_pred = lr.predict(X_test)
# So sánh
print('Giá trị thực tế: ', y_test)
print('Giá trị dự đoán: ', y_pred)

print('Bài 14: ')
# Tính R^2
r2 = r2_score(y_test, y_pred)
print('Hệ số R^2:', r2)
# Tính MSE
mse = mean_squared_error(y_test, y_pred)
print('Lỗi trung bình bình phương (MSE): ', mse)

print('Bài 15: ')
# Chỉ sử dụng tuổi (cột 0)
X_train_age = X_train[:, [0]]
X_test_age = X_test[:, [0]]
# Huấn luyện mô hình
lr_age = LinearRegression()
lr_age.fit(X_train_age, y_train)
# Dự đoán
y_pred_age = lr_age.predict(X_test_age)
print('Dự đoán với tuổi: ', y_pred_age)

print('Bài 16:')
# Lặp qua 10 đặc trưng
for i in range(10):
    X_train_single = X_train[:, [i]]
    X_test_single = X_test[:, [i]]
    lr_single = LinearRegression()
    lr_single.fit(X_train_single, y_train)
    y_pred_single = lr_single.predict(X_test_single)

    # Vẽ biểu đồ
    plt.figure()
    plt.scatter(X_test_single, y_test, color='blue', label='Thực tế')
    plt.plot(X_test_single, y_pred_single, color='red', label='Dự đoán')
    plt.xlabel(f'Đặc trưng{i + 1}')
    plt.ylabel('Tiến trình bệnh')
    plt.title(f'Hồi quy tuyến tính với đặc trưng {i + 1}')
    plt.legend()
    plt.show()

print('Bài 17: ')
# Tải bộ dữ liệu
breast_cancer = load_breast_cancer()
# In các khóa
print('Các khóa của từ điển: ', breast_cancer.keys())

print('Bài 18: ')
# Kích thước của data
print('Kích thước của dữ liệu: ', breast_cancer.data.shape)
# Chuyển target thành Series để đếm
target_series = pd.Series(breast_cancer.target)
benign_count = target_series.value_counts()[1]
malignant_count = target_series.value_counts()[0]

print('Số lượng khối u lành tính (benign):', benign_count)
print('Số lương khối u ác tính (malignant): ', malignant_count)

print('Bài 19: ')
# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2,
                                                    random_state=42)

# Đánh giá với K t 1 đến 10
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

print('Bài 20: ')
# Tạo dữ liệu
X, y = mglearn.datasets.make_forge()

# Huấn luyện và đánh giá Logistic Regression
logreg = LogisticRegression().fit(X, y)
print('Độ chính xác Logistic Regression: ', logreg.score(X, y))
# Huấn luyện và đánh giá Linear SVC
svc = LinearSVC().fit(X, y)
print('Độ chính xác Lineear SVC: ', svc.score(X, y))

print('Bài 21: ')
# Tải dữ liệu
faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# In mô tả
print('Mô tả bộ dữ liệu:\n ', faces.DESCR)

print('Bài 22: ')
print('Kích thước images: ', faces.images.shape)
print('Kích thước data: ', faces.data.shape)
print('Kích thước target: ', faces.target.shape)
print('Tên nhãn: ', faces.target_names)

print('Bài 23: ')


def plot_faces(images, n_row=2, n_col=5):
    plt.figure(figsize=(2 * n_col, 2.5 * n_row))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
    plt.show()


# Vẽ 10 khuôn mặt
plot_faces(faces.images)

print('Bài 24: ')
# Khỏi tạo SVC với kernel tuyến tính
svc = SVC(kernel='linear')

print('Bài 25: ')
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=42)
print('Kích thước tập huấn luyện: ', X_train.shape)
print('Kích thước tập kiếm tra: ', X_test.shape)

print('Bài 26: ')


def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(f'Độ chính xác K-fold (k={k}:{scores.mean():.2f}(+/-{scores.std() * 2:.2f})')


print('Bài 27: ')


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_scores = model.score(X_train, y_train)
    test_scores = model.score(X_test, y_test)
    print('Độ chính xác tập huấn luyện: ', train_scores)
    print('Độ chính xác tập kiểm tra: ', test_scores)


print('Bài 28: ')
# Sử dụng SVC vơới kernel tuyến tính
svc = SVC(kernel='linear')

# Đánh giá với cross-validation
evaluate_cross_validation(svc, faces.data, faces.target)
# Huấn luyện và đánh gái trn tập chia
train_and_evaluate(svc, X_train, X_test, y_train, y_test)

print('Bài 29: ')


# Lưu ý: Trong thực tế, bạn cần dữ liệu thực về "kính" (Chẳng hạn từ siêu dữ liệu hoặc phân tích ảnh), nhưng ở đây ta dùng giả lập để tiếp tục bài tập
# Giả lập nhãn đeo kính dựa trên chỉ số (ví dụ: cứ 2 ảnh thì 1 ảnh đeo kính)
def create_glasses_target(target):
    # Tạo mảng ngẫu nhiên: 1 cho đeo kính , 0 cho không đeo kính
    np.random.seed(42)  # Để kết quả nhất quán
    glasses_target = np.random.randint(0, 2, size=len(target))
    return glasses_target


# Tạo mảng mục tiêu mới
faces_glasses_target = create_glasses_target(faces.target)
print('Mảng mục tiêu mới (10 giá trị đầu tiên):', faces_glasses_target[:10])

print('Bài 30: ')
# Chia dữ liệu với mảng mục tiêu mới
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces_glasses_target, test_size=0.25, random_state=42)
# Tạo SVC mới với kernel tuyến tính
svc_2 = SVC(kernel='linear')
# Huấn luyện mô hình
svc_2.fit(X_train, y_train)

print('Bài 31: ')


def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(f'Độ chính xác K-fold (k={k}:{scores.mean():.2f}(+/-{scores.std() * 2:.2f})')


svc_2 = SVC(kernel='linear')
evaluate_cross_validation(svc_2, X_train, y_train, 5)

print('Bài 32: ')
# Tách tập đánh giá (10 hình ảnh từ chỉ số 30 đến 39)
X_eval = faces.data[30:40]
y_eval = faces_glasses_target[30:40]

# Tập huấn luyện là phần còn lại
X_train_remaining = np.concatenate((faces.data[:30], faces.data[:40]))
y_train_remaining = np.concatenate((faces_glasses_target[:30], faces_glasses_target[:40]))

# Tạo và huấn luyện SVC mới
svc_3 = SVC(kernel='linear')
svc_3.fit(X_train_remaining, y_train_remaining)

# Đánh giá trân tập 10 ảnh
accuracy = svc_3.score(X_eval, y_eval)
print('Độ chính xác trên tập đánh giá 10 ảnh: ', accuracy)

print('Bài 33')
# Dư đoán
y_pred = svc_3.predict(X_eval)

# Định dạng lại dữ liệu từ mảng phẳng thành ma trận 64x64
eval_faces = [np.reshape(a, (50, 37)) for a in X_eval]  # Kích thước thực tế của faces.images à 50x37


# Hàm plot_faces đã định nghĩa trước (điều chỉnh lại)
def plot_faces(images, predictions, n_col=10):
    plt.figure(figsize=(2 * n_col, 2.5))
    for i in range(len(images)):
        plt.subplot(1, n_col, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'pred:{predictions[i]}')
        plt.axis('off')
    plt.show()


# Vẽ aảnh với dự đoán
plot_faces(eval_faces, y_pred)
# Xác định ảnh sai
for i in range(len(y_eval)):
    if y_eval[i] != y_pred[i]:
        print(f'Ảnh ở chỉ ố {i + 30} bị phân loại sai. Thực tế: {y_eval[i]}, Dự đoán: {y_pred[i]}')

print('Bài 34: ')
# Tải toàn bộ tập dữ liệu 20 Newsgroups
news = fetch_20newsgroups(subset='all')

# Kiểm tra số lượng bài báo
print(f"Số lượng bài báo: {len(news.data)}")

# Xem danh sách các chủ đề (nhóm tin tức)
print(f"Các chủ đề: {news.target_names[:5]}")


print('Bài 35: ')
# Kiểm tra kiểu dữ liệu của các thuộc tính chính
print(type(news.data))          # Danh sách văn bản (list)
print(type(news.target))        # Mảng numpy chứa nhãn (numpy.ndarray)
print(type(news.target_names))  # Danh sách tên chủ đề (list)

# In danh sách các nhóm tin tức (chủ đề)
print("Danh sách các chủ đề:")
print(news.target_names)

print('Bài 36: ')
# Chia dữ liệu thành tập huấn luyện (75%) và tập kiểm tra (25%)
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=42)

# Kiểm tra kích thước tập dữ liệu sau khi chia
print(f"Số bài báo trong tập huấn luyện: {len(X_train)}")
print(f"Số bài báo trong tập kiểm tra: {len(X_test)}")

print('Bài 37: ')
def train_and_evaluate(vectorizer, X_train, X_test, y_train, y_test):
    # Biến đổi văn bản thành đặc trưng số
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Khởi tạo mô hình Naive Bayes
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test_vec)

    # Đánh giá độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
# 1. CountVectorizer
count_vectorizer = CountVectorizer()
count_acc = train_and_evaluate(count_vectorizer, X_train, X_test, y_train, y_test)
print(f"Độ chính xác với CountVectorizer: {count_acc:.4f}")

# 2. TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_acc = train_and_evaluate(tfidf_vectorizer, X_train, X_test, y_train, y_test)
print(f"Độ chính xác với TfidfVectorizer: {tfidf_acc:.4f}")

# 3. HashingVectorizer (Không cần fit, chỉ transform)
hashing_vectorizer = HashingVectorizer(n_features=2**16)  # Giới hạn số đặc trưng để giảm kích thước
hashing_acc = train_and_evaluate(hashing_vectorizer, X_train, X_test, y_train, y_test)
print(f"Độ chính xác với HashingVectorizer: {hashing_acc:.4f}")
