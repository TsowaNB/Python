# # List of days in a week
# days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# # Using a for loop to print each day
# for day in days_of_week:
#     print(day)

# # Arithmetic Operations
# a = 10
# b = 3

# print(a + b)  # Output: 13
# print(a - b)  # Output: 7
# print(a * b)  # Output: 30
# print(a / b)  # Output: 3.3333333333333335
# print(a % b)  # Output: 1
# print(a ** b) # Output: 1000
# print(a // b) # Output: 3

# # # String Operations
# # greeting = "Hello"
# # name = "Bob"

# # print(greeting + ", " + name + "!")  # Output: Hello, Bob!
# # print(greeting * 3)  # Output: HelloHelloHello


# i = 1 
# # if i < 6:
# while i < 6:
#     print(i)
#     i =+ 1

# adj = ["red", "big", "tasty"]
# fruits = ["apple", "banana", "cherry"]

# for x in adj:
#   for y in fruits:
#     print(x, y)

# class myClass:
#  x = 5
#  p1 = myClass()
#  print(p1.x)

# class MyClass:
#   x = 5

# p1 = MyClass()
# print(p1.x)

# class person:
#     def __init__(self, name, age):
#         self.name = name
# #         self.age = age
# x = person(" john" , 36)

# print(x.name)
# print(x.age)


# class person:
#     def __init__(self, firstname, lastname):
#         self.firstname = firstname
#         self.lastname = lastname

#     def printname(self):
#         print(self.firstname, self.lastname)

# x = person("joy", "doe")
# x.printname()


# def student_grades(score):
#     if 70 <= score <= 100:
#         return 'A'
#     elif 60 <= score <= 69:
#       return 'B'
#     elif 50 <= score <= 59:
#         return 'c'
#     elif 45 <= score <= 49:
#         return 'D'
#     elif 40 <= score <= 44:
#         return'D'
#     elif 0 <= score <= 39:
#         return 'F'
#     else:
#         return 
    
# grades = student_grades(0)
# print (grades)

# #  Input the Number of Subjects
# subjects_num = int(input("number of subjects "))

# # Input Grades
# grades = []
# for i in range(subjects_num):
#     grade = float(input(f"  grade for subject {i+1}: "))
#     grades.append(grade)

# #  Calculate the Average Grade
# average_score = sum(grades) / subjects_num

# #  Determine the Final Grade
# if average_score >= 70:
#     final_grade = 'A'
# elif average_score >= 60:
#     final_grade = 'B'
# elif average_score >= 50:
#     final_grade = 'C'
# elif average_score >= 45:
#     final_grade = 'D'
# else:
#     final_grade = 'F'

# print(f"\nAverage Grade: {average_score:.2f}")
# print(f"Final Grade: {final_grade}")


# Enter the number of subjects: 4
# Enter the grade for subject 1: 85
# Enter the grade for subject 2: 78
# Enter the grade for subject 3: 90
# Enter the grade for subject 4: 70

# Average Grade: 80.75
# Final Grade: A




# # Step 1: Input the Number of Subjects
# num_subjects = int(input("How many subjects do you have? "))

# # Step 2: Input Grades
# grades = [float(input(f"Enter grade for subject {i+1}: ")) for i in range(num_subjects)]

# # Step 3: Calculate the Average Grade
# average_grade = sum(grades) / num_subjects

# # Step 4: Determine the Final Grade
# if average_grade >= 70:
#     final_grade = 'A'
# elif average_grade >= 60:
#     final_grade = 'B'
# elif average_grade >= 50:
#     final_grade = 'C'
# elif average_grade >= 45:
#     final_grade = 'D'
# else:
#     final_grade = 'F'

# # Output the Results
# print(f"\nYour average grade is {average_grade:.2f}")
# print(f"Your final grade is {final_grade}")



# import math
# # Input coefficients
# a = float(input("Enter coefficient a: "))
# b = float(input("Enter coefficient b: "))
# c = float(input("Enter coefficient c: "))
# # Calculate the discriminant
# discriminant = b**2 - 4*a*c
# # Check if the discriminant is positive, negative, or zero
# if discriminant > 0:
#  # Two real and distinct roots
#  root1 = (-b + math.sqrt(discriminant)) / (2*a)
#  root2 = (-b - math.sqrt(discriminant)) / (2*a)
#  print(f"Root 1: {root1}")
#  print(f"Root 2: {root2}")
# elif discriminant == 0:
#  # One real root (repeated)
#  root = -b / (2*a)
#  print(f"Root: {root}")
# else:
#  # Complex roots
#  real_part = -b / (2*a)
#  imaginary_part = math.sqrt(abs(discriminant)) / (2*a)
#  print(f"Root 1: {real_part} + {imaginary_part}i")
#  print(f"Root 2: {real_part} - {imaginary_part}i")


# import numpy

# speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# x = numpy.mean(speed)
# print(x)

# y = numpy.median(speed)
# print(y)

# from scipy import stats
# speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# p = stats.mode(speed)
# print(p)

# import numpy

# ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
# x = numpy.percentile(ages,97)
# print(x)
# import numpy
# x = numpy.random.uniform(0.0, 5.0, 250)
# print(x)

# import numpy
# import matplotlib.pyplot as plt

# x = numpy.random.uniform(0.0, 5.0, 250)

# plt.hist(x, 5)
# plt.show()


# import qiskit
# import pennylane as qml
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import sklearn
# import SimpleITK as sitk

# print("Qiskit version:", qiskit.__version__)
# print("PennyLane version:", qml.__version__)
# print("TensorFlow version:", tf.__version__)
# print("NumPy version:", np.__version__)
# print("Pandas version:", pd.__version__)
# print("Scikit-learn version:", sklearn.__version__)
# print("SimpleITK version:", sitk.Version())

# import matplotlib.pyplot as plt
# import numpy as np
# x = np.random.uniform(1.0, 5.0,1000)

# plt.hist(x,5)
# plt.legend()
# plt.show()


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
# # print(len(x_train), len(y_train), len(x_test), len(y_test))

# # print(x_train, y_train)

# class LinearRegModelv2(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.linear_layer = nn.Linear(in_features=1,
#                                       out_features=1)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.linear_layer(x)
        
# torch.manual_seed(42)
# model_1 = LinearRegModelv2()
 

# # print(x_train[:5], y_train[:5])

# loss_fn = nn.L1Loss()
# optimizer = torch.optim.SGD(params=model_1.parameters,
#                             lr = 0.001)
# torch.manual_seed(42)
# epochs = 200
# for epoch in range(epochs):
#     model_1.train(x_train)

# y_pred = model_1(x_train)
# loss = loss_fn(y_pred, y_train)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# model_1.eval()
# with torch.inference_mode():
#     test_pred = model_1(x_test)

#     test_loss = loss_fn(test_pred, y_test)

# if epoch % 10 == 0:
#     print(f"Epoch: {epoch}, Loss: {loss}, Test loss:{test_loss}")







