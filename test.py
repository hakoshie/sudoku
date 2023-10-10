import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import random
import warnings
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from tqdm.notebook import tqdm
from sklearn.exceptions import ConvergenceWarning


# warnings.simplefilter("ignore", ConvergenceWarning)
# warnings.simplefilter("ignore", FutureWarning)
# os.environ["PYTHONWARNINGS"] = "ignore"
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC


import pickle
# save model and scaler
# # file="svc_rbf"
# file="rf"
# with open(f'./pickle/{file}_clf.pickle', mode='wb') as fp:
#     pickle.dump(clf, fp)
# with open(f'./pickle/{file}_scaler.pickle', mode='wb') as fp:
#     pickle.dump(scaler, fp)

# reload
scaler = pd.read_pickle('./pickle/rf_scaler.pickle')
clf=pd.read_pickle('./pickle/rf_clf.pickle')
# X_test = scaler.transform(X_test)
# acc_test = 100.0 * clf.score(X_test, y_test)
# print("rf_acc_test", acc_test)
cv2.imread("./data/level1/sample.jpg").shape
# import recog
import recog_l2
# import cv2
pixel=60
# cropped=recog.recognize(path="./data/level_1.jpg")
problems=recog_l2.recognize(path="./data/level1/sample.jpg")
print(problems)


# cropped_rs.shape
# plt.imshow(cropped_rs, cmap='gray')
# i,j=2,3
# def plot_ij(i,j):
#     plt.imshow(cropped_rs[i*pixel:(i+1)*pixel, j*pixel:(j+1)*pixel], cmap='gray')
# plot_ij(i,j)
# plot_ij(8,0)
# # cropped_rs
# predicted_digits=[]
# # cropped_rs=cv2.bitwise_not(cropped_rs)
# # cropped_region_rs=cv2.cvtColor(cropped_region_rs, cv2.COLOR_RGB2GRAY)
# for i in range(9):
#     for j in range(9):
#         digit_square = cropped_rs[i*pixel:(i+1)*pixel, j*pixel:(j+1)*pixel]
#         # plt.imshow(cropped_region_[i*pixel:(i+1)*pixel, j*pixel:(j+1)*pixel])
#         if np.mean(digit_square) >250:
#             # print(i,j)
#             predicted_digits.append(0)
#             continue
#         digit_square = np.array(digit_square / 255, dtype="double").reshape(1, -1)
#         digit_square=scaler.transform(digit_square)

#         # print(digit_square.shape)
#         # digit_square = np.expand_dims(digit_square, axis=0)
#         # print(digit_square.shape)
#         prediction = clf.predict(digit_square)
#         # print(prediction)
#         predicted_digit = np.argmax(prediction)
#         predicted_digits.append(prediction[0])
# problem=[]
# for i in range(0, len(predicted_digits), 9):
#     problem.append(predicted_digits[i:i+9])
#     print(predicted_digits[i:i+9])
# matrix_data = [ [0, 3, 5, 0, 9, 0, 0, 4, 8],
#                 [0, 0, 9, 0, 0, 8, 0, 0, 3],
#                 [0, 4, 0, 6, 0, 5, 0, 0, 1],
#                 [0, 0, 0, 0, 7, 4, 0, 0, 0],
#                 [0, 2, 0, 0, 0, 0, 0, 6, 0],
#                 [0, 0, 0, 1, 5, 0, 0, 0, 0],
#                 [8, 0, 0, 9, 0, 2, 0, 7, 0],
#                 [9, 0, 0, 5, 0, 0, 2, 0, 0],
#                 [6, 1, 0, 0, 4, 0, 5, 3, 0]]
# validate=[matrix_data[i][j]==problem[i][j] for i in range(9) for j in range(9)]
# proba=sum(validate)/len(validate)
# zeros=np.sum(np.array(matrix_data) == 0)
# nonzeros=81-zeros
# failed=81-proba*81
# print(proba,failed,f"failure rate: {failed/nonzeros:.2f}")

