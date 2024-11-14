# if 5 > 2:
#      print("Five is greater than two!")

# """This is a comment"""
# print("Hello, World!")

# print("Hello, World!") #This is a comment

# # x = 5
# y = "John"
# # print(type(x))
# print(y)

# x = 4 # x is of type int
# x = "Sally" # x is now of type str
# print(x)

# x = "awesome"
# print("Python is " + x)

# x = ["apple", "banana", "cherry"]
# print(type(x)) 

# x = range(6)
# print(type(x))
# print(x)

# x = {"name" : "John", "age" : 36}
# print(type(x)) 
# print(x)

# a = 33
# b = 200
# if b > a:
#  pass

# i = 1

# i += 1



# import datetime
# x = datetime.datetime.now()
# print(x)

#

# add = sum
# print(sum(x))

# import numpy
# speed = [99, 86,87, 88,111, 86, 103, 87, 94, 78, 77,85,86]
# x = numpy.mean(speed)
# print(x)



         
# lists = [1, 2, 3, 0, 9, 8, 7, 5, 6, 4,]
# def sumMean(list):
#     return sum(list)/len(list)

# print(sumMean(lists))


# def arithmeticFn(method, n1, n2):
#     if method == "divi sion":
#      return n1/n2

#     elif method == "multiplication":
#        return n1*n2

#     elif method == "substraction":
#        return n1-n2

#     elif method ==  "addition":
#        return n1+n2
    
#     else: 
#        return "this method does not exist"
    
# print(arithmeticFn("addition",8,6))


# candidate = [
# {"name":"Felix Tsowa", "party":"PDP", "votes": 0}
# {"name":"Emma kolo", "party":"ANPP", "votes": 0} 
# {"name":"Lucy Paul", "party":"CPC", "votes": 0}
# ]

# voters = [
# {"name":"Joe wood", "age":30, "occupation": "nurse"}
# {"name":"Ade Pat", "age":34, "occupation": "tailor"}
# {"name":"king stone", "age":29, "occupation": "farmer"}
# {"name":"Mary John", "age":25, "occupation": "teacher"}
# {"name":"Joy chioma", "age":30, "occupation": "chef"}
# {"name":"Rich Wealth", "age":40, "occupation": "doctor"}
# {"name":"Kolo pius", "age":19, "occupation": "student"}
# {"name":"Joel Nath", "age":50, "occupation": "Lawyer"}
# {"name":"Travis Green", "age":29, "occupation": "Engineer"}
# {"name":"Alice Kola", "age":21, "occupation": "Designer"}
# {"name":"Lois Doc", "age":25, "occupation": "Accountant"}
# {"name":"Nda koi", "age":60, "occupation": "Architect"}
# {"name":"Queen Love", "age":30, "occupation": "Journalist"}
# ]

# # CREATE A FUNTION TO VOTE
# def vote(voter_name, candidate_name):
#   voter_exists = any(voter["name"]== voter_name for voter in voters )
#   if not voter_exists:
#    print("voter '{voter_name}' is not registered")
#    return 
   

# # CHECK IF THE CANDIDATES
#   for candidate in candidate:
#   if candidate["name"] == candidate_name:
#    candidate["votes"] += 1
#    print(f"voter '{voter_name}' has voted for '(candidate_name)'.")
#    return

#  print(f"candidate '{candidate_name}' does not exist.")


# # creating a function to announce the winning candidate

# def announce_winner():
#   winner = max(candidate, key=lambda c: c["votes"])
#   print(f"the winning candidate is '{winner['name']}' from '{winner['party']}' votes.")




# # Step 1: Create the list of candidates
# candidates = [
#     {"name": "Alice Johnson", "party": "Party A", "votes": 0},
#     {"name": "Bob Smith", "party": "Party B", "votes": 0},
#     {"name": "Carol White", "party": "Party C", "votes": 0}
# ]

# # Step 2: Create the list of registered voters
# voters = [
#     {"name": "John Doe", "age": 30, "occupation": "Engineer"},
#     {"name": "Jane Roe", "age": 25, "occupation": "Doctor"},
#     {"name": "Mike Brown", "age": 45, "occupation": "Teacher"},
#     {"name": "Linda Green", "age": 22, "occupation": "Nurse"},
#     {"name": "James Black", "age": 35, "occupation": "Architect"},
#     {"name": "Patricia Blue", "age": 28, "occupation": "Lawyer"},
#     {"name": "Robert Red", "age": 32, "occupation": "Journalist"},
#     {"name": "Mary Yellow", "age": 27, "occupation": "Designer"},
#     {"name": "David Orange", "age": 40, "occupation": "Chef"},
#     {"name": "Susan Purple", "age": 33, "occupation": "Accountant"}
#  ]

# # Step 3: Create a function to vote
# def vote(voter_name, candidate_name):
#     # Check if the voter is registered
#     voter_exists = any(voter["name"] == voter_name for voter in voters)
#     if not voter_exists:
#         print(f"Voter '{voter_name}' is not registered.")
#         return

#     # Check if the candidate exists
#     for candidate in candidates:
#         if candidate["name"] == candidate_name:
#             candidate["votes"] += 1
#             print(f"Voter '{voter_name}' has voted for '{candidate_name}'.")
#             return

#     print(f"Candidate '{candidate_name}' does not exist.")

# # Step 4: Create a function to announce the winning candidate
# def announce_winner():
#     winner = max(candidates, key=lambda c: c["votes"])
#     print(f"The winning candidate is '{winner['name']}' from '{winner['party']}' with '{winner['votes']}' votes.")

# # Example usage
# vote("John Doe", "Alice Johnson")
# vote("Jane Roe", "Bob Smith")
# vote("Mike Brown", "Alice Johnson")
# vote("Linda Green", "Carol White")
# vote("James Black", "Alice Johnson")

# # Announce the winner
# announce_winner()


# cars = ['Ford', 'BMW', 'Volvo']

# cars.sort(reverse=False)
# print(cars)

# A function that returns the length of the value:
# def myFunc(e):
#   return len(e)

# cars = ['Ford', 'Mitsubishi', 'BMW', 'VW']

# cars.sort(key=myFunc)

# print(cars)def assign_faculty_to_students(students):

#STUDENT PORTAL
# students = [
#     {"name" : "Rich Staint", "email" : "richstaint1.gmail.com", "dept" : "Mathematics", "level": 400,"ID": 1},
#     {"name" : "Rich Great", "email" : "richgreat.gmail.com", "dept" : "Computer Engineering", "level" : 300, "ID" : 2},
#     {"name" : "Joe Staint", "email" : "joestaint.gmail.com.", "dept" : "Computer Engineering", "level": 200, "ID" : 3},
#     {"name" : "Kikiola Gbenga", "email" :"kikiola.gmail.com" , "dept" : "Forestry" ,"level": 100 , "ID" : 4 },
#     {"name" : "Zoe Staint", "email" : "zoestaint.gmail.com", "dept" : "Micro Biology" , "level": 500 ,"ID" : 5},
# ]

# def add_a_student(new_student):
#         return students.append(new_student)

# add_a_student({"name" : "Kolo Joy","email" : "kolojoy.gmail.com", "dept" : "chemistry","level": 400,"ID": 6})
# print(students)


# #function to promotestudents
# def promote_students():
#     for student in students:
#             student["level"] += 100
#     promote_students()
#     print(students)
            
#
# i = 1
# while i < 6:
#   print(i)
#   i += 1
   
# def my_function(food):
#   for x in food:
#     print(x)
#
# fruits = ["apple", "banana", "cherry"]
#
# my_function(fruits)

