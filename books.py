###     CLASSIFICATION OF NEURAL NETWORK
import sklearn
from sklearn.datasets import make_circles

n_samples = 1000
x, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

# print(len(x), len(y))

import pandas as pd
circles = pd.DataFrame({'x1': x[:,0],
                        'x2': x[:,1],
                        'label':y})
# print(circles.head(10))
import numpy as np
import matplotlib.pyplot  as plt
plt.scatter(x=x[:,0],
            y=x[:,1],
            c=y,
            cmap=plt.cm.RdYlBu)

# plt.legend(x,y)
# plt.show()
 
# print(x.shape, y.shape)
# print(x)

x_sample = x[0]
y_sample = y[0]
# print(f'value for one sample of x: {x_sample}, and for y:{y_sample}')
# print(f"shape for one sample of x: {x_sample.shape}, and for y:{y_sample.shape}")

##creating train and test split
import torch
# print(torch.__version__)
# print(type(x), x.dtype)

## turn data into tensors
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# print(type(x), x.dtype, type(y), y.dtype)
 

 ##### SPLIT  DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y, 
                                                    test_size = 0.2,
                                                 random_state=42)
# print(len(x_train), len(y_train), len(x_test), len(y_test))

### BUILDING A MODEL
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# class CircleModelV0(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = nn.Linear(in_features=2, out_features=5)
#         self.layer_2 = nn.Linear(in_features=5, out_features=1)
#     def forward(self, x):        
#         return self.layer_2(self.layer_1(x)) 
# model_0 = CircleModelV0()
# print (model_0)
# print(device)

# print(model_0.state_dict())

#   # import sklearn
# from sklearn.datasets import make_circles
#
# n_samples = 1000
# x, y = make_circles(n_samples,
#                     noise=0.03,
#                     random_state=42)
#
# # print(len(x), len(y))
# # print(f"First 5 samples of x:\n {x[:5]}")
# # print(f"First 5 samples of y:\n {y[:5]}")
# # print(y)
#
# import pandas as pd
# circles = pd.DataFrame({'x1': x[:,0],
#                         'x2': x[:,1],
#                         'label':y})
# print(circles.head(10))
#
# import matplotlib.pyplot  as plt
# plt.scatter(x=x[:,0],
#             y=x[:,1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
#
# plt.legend(x,y)
# plt.show()

# ### MULTICLASS DATA SET
# import sklearn.model_selection
# import torch
# from torch import nn
# import numpy
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_blobs

# NUM_CLASSES = 4
# NUM_FEATURES = 2
# RANDOM_SEED = 42

# ## creating multiclass data
# x_blob, y_blob, = make_blobs(n_samples=1000,
#                              n_features=NUM_FEATURES,
#                              centers=NUM_CLASSES,
#                              cluster_std=1.5,
#                              random_state=RANDOM_SEED
#                              )

# x_blob = torch.from_numpy(x_blob).type(torch.longtensor)
# y_blob = torch.from_numpy(y_blob).type(torch.longtensor)

# x_blob_train, y_blob_train, x_blob_test, y_blob_test = train_test_split(x_blob,
#                                                                         y_blob,
#                                                                         test_size=0.2,
#                                                                         random_state=RANDOM_SEED)

# plt.figure(figsize=(10, 7))
# plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)


# # plt.show()

# ## building a multiclass classification
# class BlobModel(nn.Module):
#     def __init__(self, input_features, out_features, hidden_units=8):
#         super().__init__()
#         self.linear_layer_stack = nn.Sequential(
#             nn.Linear(in_features=input_features, out_features=hidden_units),
#             nn.ReLU(),
#             nn.Linear(in_features=hidden_units, out_features=hidden_units),
#             nn.ReLU(),
#             nn.Linear(in_features=hidden_units, out_features=out_features)
#         )

#     def forward(self, x):
#         return self.linear_layer_stack


# model_4 = BlobModel(input_features=2,
#                     out_features=4,
#                     hidden_units=8)
# # print(model_4)

# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model_4.parameters(),
#                             lr=0.1)

# model_4.eval()
# with torch.inference_mode():
#   y_logits = model_4(x_blob_test.to(device))
# print(y_logits[:10])
# print(y_blob_test[:10])

# # y_logits_probs = torch.softmax(y_logits, dim=1)
# # print(y_logits[:5])
# # print(y_logits_probs[:5])   

# # torch.manuel_seed = 42
# # epochs = 100
# # x_blob_train, y_blob_train = x_blob_train.to(device), y_blob_train.to(device)
# # x_blob_test, y_blob_test = x_blob_test.to(device), y_blob_test.to(device)



### artecture  convlutional neural network model

 