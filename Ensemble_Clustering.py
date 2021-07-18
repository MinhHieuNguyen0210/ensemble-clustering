from sklearn.datasets import load_iris,  make_moons, make_circles
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from kemlglearn.datasets import make_blobs
from sklearn.cluster import KMeans
from kemlglearn.cluster.consensus import SimpleConsensusClustering
import numpy as np
from numpy.random import normal

import warnings
warnings.filterwarnings('ignore')

data = load_iris()['data']
labels = load_iris()['target'] #Sử dụng làm Ground Trurh

#Thư viện kemlglearn để triển khai một thuật toán đồng thuận đơn giản dựa trên ma trận co-association.

nc = 3 #Số phân cụm Kmeans
km = KMeans(n_clusters=nc) 

cons = SimpleConsensusClustering(n_clusters=nc, n_clusters_base=20, n_components=50, ncb_rand=False)
#n_clusters: số cụm,
#n_clusters_base: số cụm phân loại cơ sở, 
#n_components: số thành phần đồng thuận
#ncb_rand:True nếu chọn số lượng cụm thành phần ngẫu nhiên trong khoản [2..n_cluster]

lkm = km.fit_predict(data)
cons.fit(data)
lcons = cons.labels_

print('K-M AMI =', adjusted_mutual_info_score(labels, lkm)) 
print('SCC AMI =', adjusted_mutual_info_score(labels, lcons))

fig = plt.figure(figsize=(20,7))
ax = fig.add_subplot(131)
plt.scatter(data[:,0],data[:,1],c=labels) #load iris target
plt.title('Ground Truth')
ax = fig.add_subplot(132)
plt.scatter(data[:,0],data[:,1],c=lkm) #Vẽ tập hợp các điểm phân tán sau khi predict
plt.title('K-means')
ax = fig.add_subplot(133)
plt.scatter(data[:,0],data[:,1],c=lcons) #vẽ tập hợp các điểm phân tán sau khi áp dụng SimpleConsensusClustering
plt.title('Simple Consensus');
plt.show();

############# Make_circles from Sklearn
data, labels = make_circles(n_samples=400, noise=0.1, random_state=4, factor=0.3)
nc = 2
km = KMeans(n_clusters=nc)

cons = SimpleConsensusClustering(n_clusters=nc, n_clusters_base=10, n_components=20, ncb_rand=False)

lkm = km.fit_predict(data)
cons.fit(data)
lcons = cons.labels_

print('K-M AMI =', adjusted_mutual_info_score(labels, lkm))
print('SCC AMI  =', adjusted_mutual_info_score(labels, lcons))

fig = plt.figure(figsize=(20,7))
ax = fig.add_subplot(131)
plt.scatter(data[:,0],data[:,1],c=labels)
plt.title('Ground Truth')
ax = fig.add_subplot(132)
plt.scatter(data[:,0],data[:,1],c=lkm)
plt.title('K-means')
ax = fig.add_subplot(133)
plt.scatter(data[:,0],data[:,1],c=lcons)
plt.title('Simple Consensus');
plt.show()

############# Make moons from Sklearn
data, labels = make_moons(n_samples=250, noise=0.1)

nc = 2
km = KMeans(n_clusters=nc)

cons = SimpleConsensusClustering(n_clusters=nc, n_clusters_base=15, n_components=150, ncb_rand=False)

lkm = km.fit_predict(data)
cons.fit(data)
lcons = cons.labels_

print('K-M AMI =', adjusted_mutual_info_score(labels, lkm))
print('SCC AMI  =', adjusted_mutual_info_score(labels, lcons))

fig = plt.figure(figsize=(20,7))
ax = fig.add_subplot(131)
plt.scatter(data[:,0],data[:,1],c=labels)
plt.title('Ground Truth')
ax = fig.add_subplot(132)
plt.scatter(data[:,0],data[:,1],c=lkm)
plt.title('K-means')
ax = fig.add_subplot(133)
plt.scatter(data[:,0],data[:,1],c=lcons)
plt.title('Simple Consensus');