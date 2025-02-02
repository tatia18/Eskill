import numpy as np
import matplotlib.pyplot as plt

D = 2
K = 3
N = 300

mu1 = np.array([0, 0])
mu2 = np.array([5, 5])
mu3 = np.array([0, 5])

X = np.zeros((N, D))

X[:100, :] = np.random.randn(100, D)+ mu1
X[100:200, :] = np.random.randn(100, D)+mu2
X[200:, :] = np.random.randn(100, D)+mu3


plt.scatter(X[:, 0], X[:, 1]); #ეს ხაზი ქმნის scatter plot-ს მონაცემებისთვის, 
#სადაც X არის მატრიცა და იყენებს პირველ ორ სვეტს (0 და 1 ინდექსები) x და y კოორდინატებისთვის.

cluster_centers = np.zeros((K, D))
for k in range(K):
    i = np.random.choice(N)
    cluster_centers[k] = X[i]
#K კლასტერის ცენტრების ინიციალიზაცია D განზომილებიან სივრცეში
#თითოეული კლასტერის ცენტრისთვის შემთხვევითად ირჩევს წერტილს არსებული მონაცემებიდან (X)

max_iters = 20
cluster_identities = np.zeros(N)
# განსაზღვრავს მაქსიმალურ იტერაციების რაოდენობას (20)
# ქმნის მასივს cluster_identities, სადაც შეინახება თითოეული წერტილის კლასტერის იდენტიფიკატორი

saved_cluster_identities = [] #იქმნება ცარიელი სია წინა იტერაციების შესანახად
for i in range (max_iters):
    old_cluster_identities =  cluster_identities.copy()
    saved_cluster_identities.append(old_cluster_identities)


# კლასტერების მინიჭება ნაბიჯი 1
# თითოეული წერტილისთვის პოულობს უახლოეს კლასტერის ცენტრს
# იყენებს ევკლიდური მანძილის კვადრატს მსგავსების საზომად

for n in range(N):
    closest_k = -1
    min_dist = float('inf')
    for k in range(K):
        d = (X[n] - cluster_centers[k]).dot(X[n] - cluster_centers[k])
        if d < min_dist:
            min_dist = d
            closest_k = k
    cluster_identities[n] = closest_k


# ნაბიჯი 2 - ცენტრების გადათვლა:
for k in range(K):
    cluster_centers[k,:] = X[cluster_identities == k].mean(axis=0)

# თითოეული კლასტერისთვის გადაითვლის ცენტრს, როგორც კლასტერში შემავალი წერტილების საშუალოს

# 3) კონვერგენციის შემოწმება:

if np.all(old_cluster_identities == cluster_identities):
    print(f"Converged on step {i}")
    
# ამოწმებს თუ კლასტერების მინიჭება აღარ იცვლება

# ვიზუალიზაცია:
plt.scatter(X[:,0], X[:,1], c=cluster_identities)
plt.scatter(cluster_centers[:,0], cluster_centers[:,1], s=500, c='red', marker='*')
plt.show()
# 5)ტრენირების პროგრესის ჩვენება:
M = len(saved_cluster_identities)
fig, ax = plt.subplots(figsize=(5, 5 * M))
for i in range(M):
    plt.subplot(M, 1, i + 1)
    Y = saved_cluster_identities[i]
    plt.scatter(X[:,0], X[:,1], c=Y)
    # აჩვენებს კლასტერიზაციის პროცესს თითოეული იტერაციისთვის

# plt.show() jypetshi xazavs aq ar achvenebs ratongac

