# import pennylane as qml
# print("PennyLane version:", qml.__version__)
#
# import qiskit
#
#
# print("Qiskit version:", qiskit.__version__)

# import pennylane as qml
# dev = qml.device('defauly.qubit', wires = 2)
# def qc():
#     qml.PauliX(wires = 0)
#     qml.Hadamard(wires = 0)
#     return qml.state()
# qcirc = qml.QNode(qc, dev) # Assemble the circuit & the device.
# qcirc() # Run it!

# from qiskit.quantum_info import Statevector
# zero = Statevector([1,0])
# print("zero is", zero)
#
# from qiskit.quantum_info import Pauli
# Z0Z1 = Pauli("ZZI")
# print("Z0Z1 is",Z0Z1)
# print("And its matrix is")
# print(Z0Z1.to_matrix())


# from qiskit.algorithms.minimum_eigensolvers import \
# NumPyMinimumEigensolver
# solver = NumPyMinimumEigensolver()
# result = solver.compute_minimum_eigenvalue(qhamiltonian)
# print(result)


# import pennylane as qml
# from pennylane import numpy as np
# seed = 1234
# np.random.seed(seed)
# symbols = ["H", "H"]
# coordinates = np.array([0.0, 0.0, -0.6991986158, 0.0, 0.0, 0.6991986158])
# H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
# print("Qubit Hamiltonian: ")
# print(H)


# import numpy as np
# from scipy.optimize import minimize
# theta = np.array(np.random.random(4*nqubits), requires_grad=True)
# result = minimize(energy, x0=theta)
# print("Optimal parameters", result.x)
# print("Energy", result.fun)

# from sklearn import model_selection, datasets, svm
# iris = datasets.load_iris()
# x = iris.data[0:100]
# y = iris.target[0:100]
# x_train, x_test,

# import numpy as np
# seed = 128
# np.random.seed(seed)

# from sklearn.datasets import make_classification

# data, labels = make_classification(n_samples =2500,
# n_features = 2, n_informative = 2, n_redundant =0,
# weights = (0.2, 0.8), class_sep = 0.5, random_state = seed)



# from sklearn.model_selection import train_test_split
# # Split into a training and a test dataset.
# x_train, x_test, y_train, y_test = train_test_split(
#     data, labels, shuffle = True, train_size = 0.8)
# # Split the test dataset to get a validation one.
# x_val, x_test,y_val, y_test = train_test_split(
#      x_test, y_test, shuffle = True, train_size = 0.5)

# import tensorflow as tf
# tf.random.set_seed(seed)
# modelxs_tr, y_tr) = tf.keras.Sequential([
# tf.keras.layers.Input(2),
# tf.keras.layers.Dense(8, activation = "elu"),
# tf.keras.layers.Dense(16, activation = "elu"),
# tf.keras.layers.Dense(8, activation = "elu"),
# tf.keras.layers.Dense(1, activation = "sigmoid"),
# ])
# opt = tf.keras.optimizers.Adam()
# lossf = tf.keras.losses.BinaryCrossentropy()
# model.compile(optimizer = opt, loss = lossf)

# history = model.fit(x_train, y_train,
#     validation_data = (x_train, y_train), epoch = 8,
#    batch_size = None)


# ## receivers operating characteristic curve(roc curve)
# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# fpr, tpr, _ = roc_curve(y_test, output)
# plt.plot(fpr, tpr)
# plt.plot([0,1],[0,1],linestyle="--",color="black")
# plt.xlabel("FPR"); plt.ylabel("TPR")
# plt.show()

# import numpy as np
# seed = 1234
# np.random.seed(seed)

# from sklearn.datasets import load_wine
# x,y = load_wine(return_x_y = True)

# x = x[:59+71]
# y = y[:59+71]

# from sklearn.model_selection import train_test_split
# x_tr, x_test, y_tr, y_test = train_test_split(x, y, train_size= 0.9)

# from sklearn.preprocessing import MaxAbsScaler
# scaler = MaxAbsScaler()
# x_tr = scaler.fit_transform(x_tr)

# x_test = scaler.transform(x_test)
# x_test = np.clip(x_test, 0, 1)

# import pennylane as qml
# nqubits = 4
# dev = qml.device("lightning.qubit", wires = nqubits)
# @qml.qnode(dev)
# def kernel_circ(a, b):
#  qml.AmplitudeEmbedding(
#  a, wires=range(nqubits), pad_with=0, normalize=True)
#  qml.adjoint(qml.AmplitudeEmbedding(
#  b, wires=range(nqubits), pad_with=0, normalize=True))
#  return qml.probs(wires = range(nqubits))

# from sklearn.svm import SVC
# def qkernel(A, B):
#   return np.array([[kernel_circ(a, b)[0] for b in B] for a in A])
# svm = SVC(kernel = qkernel).fit(x_tr, y_tr)

#  from sklearn.metrics import accuracy_score
# print(accuracy_score(svm.predict(x_test), y_test))mp


# import pennylane as qml
# from pennylane import numpy as np
# from qiskit.visualization import plot_histogram
# import matplotlib.pyplot as plt
# dev = qml.device('default.qubit', wires = 2)

# @qml.qnode(dev)
# def simple_cir(x):
#     qml.Hadamard(wires = 0)
#     qml.CNOT(wires=[0,1])
#     qml.RX(x[0], wires= 1)
#     return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# x = np.array([np.pi/2])
# output = simple_cir(x)
# print(f"Output of the quantum circuit: {output}")



# Install necessary libraries
# pip install pennylane pennylane-qiskit

# import pennylane as qml
# import numpy as np

# # Step 1: Define the quantum device (simulator)
# dev = qml.device("default.qubit", wires=2)

# # Step 2: Define the variational circuit
# @qml.qnode(dev)
# def variational_circuit(weights):
#     # Apply a layer of rotation gates (parametrized by weights)
#     qml.RX(weights[0], wires=0)
#     qml.RY(weights[1], wires=1)

#     # Entangling the qubits
#     qml.CNOT(wires=[0, 1])

#     # Measure the expectation value of PauliZ on the first qubit
#     return qml.expval(qml.PauliZ(0))

# # Step 3: Define a simple cost function
# def cost(weights):
#     return variational_circuit(weights)

# # Step 4: Initialize weights randomly and optimize
# weights = np.random.random(2)
# optimizer = qml.GradientDescentOptimizer(stepsize=0.4)

# # Perform optimization
# for i in range(50):
#     weights = optimizer.step(cost, weights)
#     print(f"Step {i+1}, Cost: {cost(weights)}")

# print("Optimized weights:", weights)



# import pennylane as qml
# from pennylane import numpy as np

# # Define a quantum device (simulator)
# dev = qml.device("default.qubit", wires=2)

# # Define a quantum function to be a QNode
# @qml.qnode(dev)
# def quantum_circuit(params):
#     # Apply a rotation on the first qubit
#     qml.RX(params[0], wires=0)
#     # Apply a Hadamard gate on the second qubit
#     qml.Hadamard(wires=1)
#     # Apply a controlled NOT (CNOT) gate
#     qml.CNOT(wires=[0, 1])
#     # Return measurement probabilities
#     return qml.probs(wires=[0, 1])

# # Define input parameters (angles for the rotation)
# params = np.array([0.5])

# # Execute the QNode (quantum circuit)
# probabilities = quantum_circuit(params)
# print(probabilities)



# import pennylane as qml
# from pennylane import numpy as np
# from scipy.optimize import minimize

# # Device initialization
# dev = qml.device("default.qubit", wires=2)

# # Define QNode
# @qml.qnode(dev)
# def quantum_circuit(params):
#     qml.RX(params[0], wires=0)
#     qml.RY(params[1], wires=1)
#     qml.CNOT(wires=[0, 1])
#     return qml.expval(qml.PauliZ(0))  # Expectation value of Pauli-Z on the first qubit

# # Loss function for optimization
# def cost(params):
#     return quantum_circuit(params)

# # Initial parameters
# init_params = np.array([0.5, 0.3])

# # Perform optimization
# opt_result = minimize(cost, init_params, method="BFGS")

# print("Optimized parameters:", opt_result.x)
# print("Minimum cost:", opt_result.fun)


# from qiskit_aer.aerprovider import AerSimulator
# from qiskit import QuantumCircuit, execute
# from qiskit_aer import Aer
# from qiskit import *
# # Create a quantum circuit with 1 qubit
# qc = QuantumCircuit(1)

# # Apply Pauli gates
# qc.x(0)  # Pauli-X (NOT gate)
# qc.y(0)  # Pauli-Y gate
# qc.z(0)  # Pauli-Z gate

# # Use Aer simulator to run the circuit
# backend = Aer.get_backend('statevector_simulator')
# job = execute(qc, backend)
# result = job.result()

# # Get and print the final statevector
# statevector = result.get_statevector()
# print("Final statevector:", statevector)

# # # Visualize the circuit
# qc.draw('mpl')

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import cross_val_score

# def lda(x_train, x_test, y_train, y_test):

#     clf = LinearDiscriminantAnalysis()
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     print("\n")
    # print("Linear Discriminant Analysis Metrics")

# import os
# import numpy as np
# import pandas as pd
# data = {'Size': ['small', 'small', 'large', 'medium', 'large', 'large', 'small',
#         'medium'],
#         'Color': ['red', 'green', 'black', 'white', 'blue', 'red', 'green', 'black'],
#         'Class': [1, 1, 1, 0, 1, 0, 0, 1]}
# df = pd.DataFrame(data, columns = ['Size', 'Color', 'Class'])
# print(df)

# from sklearn.preprocessing import OrdinalEncoder
#  # Creating an instance of OrdinalEncoder
# enc = OrdinalEncoder()
#  # Assigning numerical values and storing it
# enc.fit(df[["Size","Color"]])
# df[["Size","Color"]] = enc.transform(df[["Size","Color"]])
#  # Display Dataframe
# # print(df)
# df= pd.get_dummies(df, prefix="One",columns=['Size', 'Color'])
# # print(df)

# from sklearn.preprocessing import OneHotEncoder
# enc =OneHotEncoder(handle_unknown='ignore')
# enc_df =pd.DataFrame(enc.fit_transform(df[['Size','Color']]).toarray())
# df= df.join(enc_df)
# print(df)

#
# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt
# import numpy as np
# digits = load_digits()
# X = digits.data
# y = digits.target
# n_samples = 1797
# n_features = 64
# n_neighbors = 30
#  # Plot an extract of the data
# fig, axs = plt.subplots(nrows=20, ncols=20, figsize=(10, 10))
# for idx, ax0 in enumerate(axs.ravel()):
#  # cmap=plt.cm.binary to display data in black (digits) and white (background)
#     ax0.imshow(X[idx].reshape((8, 8)), cmap=plt.cm.binary)
#     ax0.axis("off")
# extract = fig.suptitle("Digits Dataset Selection", fontsize=15)


# #
# import pennylane as qml
# from pennylane import numpy as np

# dev = qml.device('default.qubit', wires=1)

# @qml.qnode(dev)
# def quantum_node(param):
#     qml.Rx(param[0], wire = [0])
#     return qml.expval(qml.PauliZ(0))
# def cost_function(param):
#     return quantum_node(param)
# param = np.array_from_array([10])
# opt = qml.GradientDescentOptimizer(stepsize= 0.4)
# steps = 100

# for i in range(steps):
#     param = opt.step(cost_function, param)
# if (i + 1) % 10 == 0:
#     print(f"Step {i+1}, Cost: {cost_function(param)}, Params: {param}")










# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# from torch import nn
# import matplotlib as plt

# weight = 0.7
# bias = 0.3

# start = 0
# end = 1
# step = 0.02
# x = torch.arange(start, end, step).unsqueeze(dim=1)
# y = weight*x + bias
# # print(x[:10], y[:10])
# # print(len(x), len(y))

# ### splitting a datasets into train and test
# train_split = int(0.8 * len(x))
# x_train, y_train = x[:train_split], y[:train_split]
# x_test, y_test = x[train_split:], y[train_split:]
# print(len(x_train), len(y_train), len(x_test), len(y_test))

# print(x_train, y_train)



# def plot_predictions(train_data = x_train ,
#              train_labels = y_train,
#              test_data = x_test,
#              test_labels = y_test,
#              predictions = None):
    
#     plt.figure(figsize = (10, 7))
#     plt.scatter(train_data,  train_labels, c = "b", s = 4, label = "Training data")
#     plt.scatter(test_data,  test_labels, c = "g", s = 4, label = "testing data")

# plot_predictions(x_train, y_train, x_test, y_test)
# plt.legend(prop = {"size": 14})
# plt.show


# import torch
# from torch import nn
# class LinearRegressionModel(nn.Module):
    # def _intit_(self):
    #     super().__init__()
#         self.weight = nn.parameter(torch.randn(1,
#                                                requires_grad= True,
#                                                dtype=torch.float))

#         self.bias = nn.parameter(torch.randn(1,
#                                                requires_grad= True,
#                                                dtype=torch.float))
        
# def forward(self, x: torch.Tensor) -> torch.Tensor:
#      return self.weight * x + self.bias 
# torch.manual_seed(42)
# model = LinearRegressionModel
# # print(x_test, y_test) 

# with torch.inference_mode():
#     y_pred =  model(x_test)

# print( y_pred)
# print(list(model.parameters()))

# # loss_fn = nn.L1Loss
# # optimizer = torch.Optim.SGD(params=model.parameters(),
# #                             lr=0.01)


# import pennylane as qml
# import numpy as np
# import tensorflow as tf
 

# seed = 1234
# np.random.seed(seed)
# tf.random.set_seed(seed)
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split 
# from sklearn.datasets import make_classification 
# tf.keras.backend.set_floatx('float64')

# import pennylane as qml
# state_0 = [[1], [0]]
# M = state_0 * np.conj(state_0).T
# import matplotlib.pyplot as plt

# def plot_losses(history):
#     tr_loss = history.history["loss"]
#     val_loss = history.history["val_loss"]
#     epochs = np.array(range(len(tr_loss))) + 1
#     plt.plot(epochs, tr_loss, label = "Training loss")
#     plt.plot(epochs, val_loss, label = "Validation loss")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()

# x, y = make_classification(n_samples = 1000, n_features = 20)

# x_tr, x_test, y_tr, y_test = train_test_split(
# x, y, train_size = 0.8)
# x_val, x_test, y_val, y_test = train_test_split(
# x_test, y_test, train_size = 0.5)

# def TwoLocal(nqubits, theta, reps = 1):
#     for r in range(reps):
#         for i in range(nqubits):
#             qml.RY(theta[r * nqubits + i], wires = i)

#         for i in range(nqubits - 1):
#             qml.CNOT(wires = [i, i + 1])

#     for i in range(nqubits):
#         qml.RY(theta[reps * nqubits + i], wires = i)


# nqubits = 4
# dev = qml.device("lightning.qubit", wires = nqubits)
# @qml.qnode(dev, interface="tf", diff_method = "adjoint")
# def qnn(inputs, theta):
#     qml.AngleEmbedding(inputs, range(nqubits))
#     TwoLocal(nqubits, theta, reps = 2)
#     return qml.expval(qml.Hermitian(M, wires = [0]))       
# weights = {"theta": 12}
  
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Input(20),
#     tf.keras.layers.Dense(4, activation = "sigmoid"),
#     qml.qnn.KerasLayer(qnn, weights, output_dim=1)
# ])

# earlystop = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=2, verbose=1,
#     restore_best_weights=True)

# opt = tf.keras.optimizers.Adam(learning_rate = 0.005)
# model.compile(opt, loss=tf.keras.losses.BinaryCrossentropy())

# history = model.fit(x_tr, y_tr, epochs = 50, shuffle = True,
#     validation_data = (x_val, y_val),
#     batch_size = 10,
#     callbacks = [earlystop])
# plot_losses(history)

# tr_acc = accuracy_score(model.predict(x_tr) >= 0.5, y_tr)
# val_acc = accuracy_score(model.predict(x_val) >= 0.5, y_val)
# test_acc = accuracy_score(model.predict(x_test) >= 0.5, y_test)
# print("Train accuracy:", tr_acc)
# print("Validation accuracy:", val_acc)
# print("Test accuracy:", test_acc)


