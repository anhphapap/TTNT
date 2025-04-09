from cProfile import label
from re import X

import numpy as np
import matplotlib.pyplot as plt
import scatter
from matplotlib.pyplot import title
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score

# 1/ As a working example, in this section we will use a dataset of handwritten digits digitalized in matrices of 8x8 pixels,
# so each instance will consist initially of 64 attributes. We start by loading our dataset and print it’s key.
digits = load_digits()
X = digits.data
y = digits.target
print("Dataset key: ", digits.keys())

# 2/ We will use the data matrix that has the instances of 64 attributes each and the target vector that has the corresponding digit number.
# Let us define a function for printing the digits to take a look at how the instances will appear.
def plot_digits(data,labels,num_rows=2,num_cols=5,title='chữ số viết tay'):
   fig,axes = plt.subplots(num_rows,num_cols,figsize=(10,4))
   for i, ax in enumerate(axes.ravel()):
       if i < len(data):
           ax.imshow(data[i].reshape(8, 8), cmap="grey")
           ax.set_title(f"Nhan: {labels[i]}")
           ax.axis('off')
   plt.suptitle(title)
   plt.tight_layout()
   plt.show()


print("Hien thi 10 chu so dau tien")
plot_digits(X,y)



# 3/ Define a function that will plot a scatter with the two-dimensional points that will be obtained by a PCA transformation.
# Our data points will also be colored according to their classes.
def plot_pca_scatter(X, y, title="PCA của tập dữ liệu chữ số viết tay"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0],X_pca[:,1], c = y, cmap='tab10', alpha=0.6, edgecolors='w')
    plt.colorbar(scatter, label='Lớp chữ số/cụm')
    plt.title(title)
    plt.xlabel('Thành phần chính 1')
    plt.ylabel('Thành phần chính 2')
    plt.grid(True)
    plt.show()
    return X_pca, pca

print("\nVẽ biểu đồ phân tán 2D sau khi áp dụng PCA (dữ liệu gốc):")
X_pca, pca_model = plot_pca_scatter(X, y)

# 5/ We will plot all the components in the same shape as the original data (digits).
def plot_pca_components():
    pca = PCA(n_components=10)
    pca.fit(X)
    components = pca.components_

    fig, axes = plt.subplots(2,5,figsize=(10,4))
    for i, ax in enumerate(axes.ravel()):
        if i < len(components):
            ax.imshow(components[i].reshape(8,8), cmap='gray')
            ax.set_title(f'Thành phần {i+1}')
            ax.axis('off')

    plt.suptitle("10 thành phần chính dưới dạng hình ảnh 8x8")
    plt.tight_layout()
    plt.show()

print("\nHiển thị 10 thành phần PCA đầu tiên dưới dạng hình ảnh 8x8:")
plot_pca_components()


# 6/ We will show in this section how k-means works using a motivating example, the problem of clustering handwritten digits.
# So, let us first import our dataset into our Python environment and show how handwritten digits look.

# 7/ As usual, we must separate train and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nKích thước tập huấn luyện (X_train):", X_train. shape)
print ("Kích thước tập kiếm tra (X_test):", X_test. shape)


# 8/ The implementation of k-means in scikit-learn already does this (the n-init parameter allows us to establish how many
# different centroid configurations the algorithm will try). It also allows us to specify that the initial centroids will
# be sufficiently separated, leading to better results. Let's see how this works on our dataset.
def experiment_with_n_init(X_train, y_train, n_clusters=10):
    n_init_values = [1, 5, 10, 20]
    ari_scores = []
    print ("\nThử nghiệm với các giá trị ninit khác nhau cho k-means:")
    for n_init in n_init_values:
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
        kmeans.fit(X_train)
        cluster_labels = kmeans.labels_
        ari = adjusted_rand_score(y_train, cluster_labels)
        ari_scores. append(ari)
        print(f"n_init = (n_init), Chi số Adjusted Rand Index: {ari:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(n_init_values, ari_scores, marker='o', linestyle='-', color='b')
    plt.xlabel ('Giá tri n_init' )
    plt.ylabel('Chi số Adjusted Rand Index')
    plt.title( 'Ảnh hưởng của n_init đến hiệu suất phân cụm k-means')
    plt.grid(True)
    plt.show()
experiment_with_n_init(X_train, y_train)


# 9/ If we print the value of the labels_ attribute of the classifier, we get a list of the cluster numbers associated to each training instance.
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans.fit(X_train)
train_cluster_labels = kmeans.labels_
print("\nNhãn cụm của dữ liệu huấn luyện (20 mẫu đầu tiên) :")
print(train_cluster_labels[:20])
print ("\nHiền thị một số chữ số huãn luyện với nhãn cụm dự đoán:")
plot_pca_scatter(X_train, train_cluster_labels, title="Chữ số huấn luyện với nhãn cụm dự đoán")

# 10/ To predict the clusters for training data, we use the usual predict method of the classifier.
predicted_train_labels = kmeans.predict(X_train)
print("\nNhãn cụm dự đoán cho dữ liệu huãn luyện (20 mẫu đầu tiên, dùng predict):")
print(predicted_train_labels[:20])
# Kết quả sẽ giống với train_cluster_labels vì dữ liệu đã được huãn luyện

# 11/ Define a print_cluster function and show ten images from each cluster.
def print_cluster(X, cluster_labels, n_clusters=10, images_per_cluster=10):
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            selected_indices = cluster_indices[:min(images_per_cluster, len(cluster_indices))]
            selected_images = X[selected_indices]
            selected_labels = cluster_labels[selected_indices]
            num_rows = 2
            num_cols = 5
            print(f"\nCum {cluster} (hiến thị {len(selected_indices)} hình ảnh):")
            plot_digits(selected_images, selected_labels, num_rows, num_cols, title=f"Chữ số trong cụm {cluster}")
        else:
            print(f"\nCụm {cluster} trống.")
print("\nHiễn thị 10 hình ảnh từ mỗi cụm (dữ liệu huấn luyện):")
print_cluster(X_train, train_cluster_labels)


# 12/How can we evaluate our performance? Precision and all that stuff does not work, since we have no target classes to compare with.
# To evaluate, we need to know the "real" clusters, whatever that means. We can suppose, for our example, that each cluster includes every
# drawing of a certain number, and only that number. Knowing this, we can compute the adjusted Rand index between our cluster assignment and
# the expected one. The Rand index is a similar measure for accuracy, but it takes into account the fact that classes can have different names
# in both assignments. That is, if we change class names, the index does not change. The adjusted index tries to deduct from the result coincidences
# have occurred by chance. When you have the exact same clusters in both sets, the Rand index equals one, while it equals zero when there are no clusters sharing a data point.
test_cluster_labels = kmeans.predict(X_test)
train_ari = adjusted_rand_score(y_train, train_cluster_labels)
test_ari = adjusted_rand_score(y_test, test_cluster_labels)
print("\nBước 12: Đánh giá hiệu suất phân cụm băng Adjusted Rand Index (ARI):")
print(f"Chi sõ Adjusted Rand Index cho tập huãn luyện: (train_ari: .4f)")
print(f"Chi sõ Adjusted Rand Index cho tập kiếm tra: {test_ari: .4f}")
print("\nGiải thích vẽ Adjusted Rand Index (ARI): ")
print("Chi sõ Adjusted Rand Index đo lường độ tương đồng giữa các cụm dự đoán và nhãn thực tẻ.")
print("Nó có giá trị từ -1 đến 1, trong đó 1 là phần cụn hoàn hảo, ở là phần cụn ngẫu nhiên, và giá trị âm là phân cụm tệ hơn ngẫu nhiên.")


# 13/ If we want to graphically show how k-means clusters look like, we must plot them on a two-dimensional plane.
# We have learned how to do that in the previous section: Principal Component Analysis (PCA). Let's construct a meshgrid
# of points (after dimensionality reduction), calculate their assigned cluster, and plot them.
def plot_kmeans_decision_boundary(X, y, kmeans, title="Vung phần cụm k-means trong không gian 2D"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Tạo lưới (meshgrid) các điểm trong không gian 2D
    h = 0.5  # Bước của lưới
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Chuyển các điểm lưới vẽ không gian gốc (2D) để dự đoán cụm
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_labels = kmeans.predict(pca.inverse_transform(mesh_points))  # Dự đoán nhãn cụm

    # Vẽ vùng phân cụm
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, mesh_labels.reshape(xx.shape), cmap='tab10', alpha=0.3)

    # Vẽ các điểm dữ liệu thực tế
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolors='k', alpha=0.6)

    plt.title(title)
    plt.xlabel('Thanh phan chinh 1')
    plt.ylabel('Thanh phan chinh 2')
    plt.colorbar(scatter)
    plt.show()

print("\nBước 13: Vẽ vùng phân cụm k-means trong không gian 2D:")
plot_kmeans_decision_boundary(X_train, train_cluster_labels, kmeans, title="Vùng phần cụm k-means trên dữ liệu huãn luyện")






