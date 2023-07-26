import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(os.path.join(os.getcwd(), "LARES", "Good", "T-40411", "T-40411a.JPG"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
hist = cv2.reduce(threshed,1, cv2.REDUCE_AVG).reshape(-1)

th = 2
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]

img_num = 1
for i in range(len(uppers)):
    cv2.imwrite('so_splited_imgs_' + str(img_num) + '.jpg', img[uppers[i]:lowers[i],:])
    img_num += 1