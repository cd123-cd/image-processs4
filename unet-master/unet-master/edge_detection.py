#K-means均值聚类
Z = img1.reshape((-1, 3))
Z = np.float32(Z)      #转化数据类型
c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 4   #聚类中心个数，一般来说也代表聚类后的图像中的颜色的种类
ret, label, center = cv2.kmeans(Z, k, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
img9 = res.reshape((img1.shape))
cv2.namedWindow("W2")
cv2.imshow("W2", img9)
cv2.waitKey(delay = 0)