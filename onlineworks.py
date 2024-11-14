# print(" Tsowa Blessing")
# print("|||")
# full_name = ' John Joe'

# name = input(' what is your name?') 
# print( 'Hi' + name)

# birth_year = input('Birth year' )

# age = 2024 - int(birth_year)
# print(age)

# course = ("python for beginners")
# print(course.find('g'))

# x = ["i love you JESUS"]

# y = x * 100
# print(y)

# x = 3.5
# print(round(x))

# import math 
# print (math.ceil(2.9))
# print (math.floor(2.9))

# import numpy
# import math
# print(math.floor(3.9))
# course = 'Python for Beginners'
# print(course.find('n'))
# print(course.title())

# price = 1000000
# good_credit = True
# if good_credit:
#     pay = 0.1 * price
# else:
#     pay = 0.2* price 
# print(f'pay: ${ pay}')



# logical operators are or, or not, and, and not
# Guess = 1
# Guess = 2
# Guess = 3

# for x in range(4):
    # for y in range(3):
        # print(f'{x}, {y}')

# numbers = [5, 2, 5, 2, 2]
# for x in numbers:
#     print('x' * x)

# numbers = [5, 2, 5, 2, 2]
# for x_count in numbers:
#     output = ''
#     for y in range(x_count):
#         output += 'x'
#     print(output)


# numbers = [2, 2, 2, 2, 5]
# for x_count in numbers:
#     output = ''
#     for y in range(x_count):
#         output += 'x'
#     print(output)


# numbers = [30, 40, 10, 22, 8, 10, 8]
# min = numbers[0]
# for number in numbers:
#     if number < min:
#         min = number
# print(min)

# #2dimension lists matrix
# matrix = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]
# for rows in matrix:
#     for numbers in rows:
#         print(numbers)

# list function or methods

# def fruits_thanks():
#     print('thanks for the fruits boss man \n i really appreciate')


# print('hello, good morning boss')
# fruits_thanks()
# print('nice day boss')

# def greet_users( first_name, last_name ):
#     print(f' hi {first_name} {last_name}!')
#     print( 'welcome' )

# print('start')
# greet_users("John", "kolo")
# greet_users('Grace', 'Gana')
# print('finish')

# #keyword argument (position)
# def greet_users( first_name, last_name ):
#     print(f' hi {first_name} {last_name}!')
#     print( 'welcome' )

# print('start')
# greet_users(last_name = "kolo", first_name = "John" )
# greet_users(first_name = 'Grace', last_name =  'Gana')
# print('finish')


# def square(number):
#     return number * number
# print(square(3)) 


# def emojis_converter(message):
#     words = message.split(" ")
#     emojis = {
#         ":)" : "😀",
#         ":(" : "😞"
#     }
#     output = " "
#     for word in words:
#         output += emojis.get(word, word) + ""
#     return output


# message = input(" > ")
# print(emojis_converter(message))



# age = int(input('Age: '))
# print(age)

# try:
#     age = int(input('Age: '))
#     print(age)
# except ValueError:
#     print('Invalid value')


# # classes in python
# class Point:
#     def move(self):
#         print("move")

#     def draw(self):
#         print("draw")


# point1 = Point()
# point1.x = 10
# point1.y = 24
# print(point1.x)
# point1.draw()
 
 ##constructor
# class Point:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


# point1 = Point(10, 23)
# print(point1.x)

# class Person:
#     def __init__(self, name):
#         self.name = name
#     def talk(self):
#         print(f'Hi, i am {self.name}') 


# # john = Person(" John Smith")
# # print(john.name)
# # john.talk()

# john = Person(" John Smith")
# john.talk()

# bob = Person(" Josh Bob")
# bob.talk()

# ##inheritance: method of reusing code in python
# class Mammal:
#     def walk(self):
#         print("walk")


# class Dog(Mammal):
#     def bark():
#         print("bark")

# class Cat(Mammal):
#     pass

# dog1 = Dog()
# dog1.bark
# dog1.walk()

# ## Modules are file with codes
# import converters
# print(converters.kg_to_lbs(70))

##generating random numbers

# import random
# for i in range(3):
#     # print(random.random())
#     print(random.randint(10, 20))

# import random

# members = ['Ble', 'Josh', 'Joyce', 'Leah']
# print(random.choice(members))

# from pathlib import Path

# path = Path()
# for file in path.glob('*'):
#     print(file)



## excel spreead sheet on python


# #input the number of subject
# subjects_number = int(input("number of subjects : "))


# score = []
# for number in range(subjects_number):
#     grade = float(input(f" score for the subjects "))
#     score.append(grade)

# # calculation of average grade
# average_score = sum(score) / subjects_number


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

# print(f"Average Grade: {average_score}")
# print(f"Final Grade: {final_grade}")


    
# subjects_num = int(input("Enter Number of Subjects: "))
# def grade_calculator(subject_num):
#     score = {}
#     for i in range(subject_num):
#         subject_name = str(input("Enter name of subject: "))
#         grade = int(input(f"Enter subject score for {subject_name}: "))
#         score.update({subject_name: grade})
#     average = sum(score.values()) / subjects_num

#     # GRADE FOR EACH SUBJECTS
#     for i, j in score.items(): #i is for subject, j is for grade
#         if j >= 70:
#             print(f"{i}: A")
#         elif j >= 60:
#             print(f"{i}: B")
#         elif j >= 50:
#             print(f"{i}: C")
#         elif j >= 45:
#             print(f"{i}: D")
#         else:
#             print(f"{i}: F")
#  # OVERALL GRADE
#     if average >= 70:
#         overall_grade = "A"
#     elif average >= 60:
#         overall_grade = "B"
#     elif average >= 50:
#         overall_grade = "C"
#     elif average >= 40:
#         overall_grade = "D"
#     else:
#         overall_grade = "F"

#     print(f"Average Score: {average} \n"
#           f"Overall Grade: {overall_grade}")


# grade_calculator(subjects_num)


