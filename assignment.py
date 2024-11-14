# print("hello world")

# print(1+2)

# print(2**2)

# print(5//2)
# print(5%2)

# print("what's up")

# how to use variable
# red_bucket1 = "kevin"
# red_bucket = 8

# print(type(red_bucket1))


# how to use condition statement (true or false)
# print(5==4)
# print(5!=4)

# thomas_age = 3
# thomas_age = 10
# age_at_kindergarten = 5

# print(thomas_age == age_at_kindergarten)

# print("hello world")

# print(1+2)

# print(2**2)

# print(5//2)
# print(5%2)

# print("what's up")

# # how to use variable
# red_bucket1 = "kevin"
# red_bucket = 8

# print(type(red_bucket1))


# how to use condition statement (true or false)
# print(5==4)
# print(5!=4)

# thomas_age = 3
# thomas_age = 5
# age_at_kindergarten = 5

# # print(thomas_age == age_at_kindergarten)

# if thomas_age < age_at_kindergarten:
#     print("thomas should be in pre-school")
# elif thomas_age == age_at_kindergarten:
#     print("enjoy kindergarten")
# else:
#     # print("thomas should be in kindergarten or another class")
#  print("thomas should be in another class")


# # how to use funtion in python

# print("kevin stravert has a great channel")


# #STUDENT PORTAL
# students = [
#     {"name" : "Rich Staint", "email" : "richstaint1.gmail.com", "dept" : "Mathematics", "level": 100,"ID": 1},
#     {"name" : "Rich Great", "email" : "richgreat.gmail.com", "dept" : "Computer Engineering", "level" : 300, "ID" : 2},
#     {"name" : "Joe Staint", "email" : "joestaint.gmail.com.", "dept" : "Computer Engineering", "level": 200, "ID" : 3},
#     {"name" : "Kikiola Gbenga", "email" :"kikiola.gmail.com" , "dept" : "Forestry" ,"level": 100 , "ID" : 4 },
#     {"name" : "Zoe Staint", "email" : "zoestaint.gmail.com", "dept" : "Micro Biology" , "level": 500 ,"ID" : 5},
#     {"name" : "Richard Joy", "email" : "richardjoy.gmail.com", "dept" : "statistic" , "level": 100 ,"ID" : 6 },
#     {"name" : "Prince Alpha", "email" : "princealpha.gmail.com", "dept" : "Forestry" , "level": 400 , "ID" : 7},
#     {"name" : "Olaolu wealth", "email" : "olaoluwealth.gmail.com", "dept" : "Crop Production"  , "level": 200 , "ID" : 8 },
#     {"name" : "Kolo Ramat", "email" : "koloramat.gmail.com", "dept" : "Mechanical Engineering" , "level": 500 , "ID" : 9 },
#     {"name" : "Chukwudi Ada", "email" : "adachukwudi.gmail.com", "dept" : "Telecom Engineering" , "level": 300 , "ID" : 10},
#     {"name" : "Uche Queen", "email" : "uchequeen.gmail.com", "dept" : "Physics" , "level": 100, "ID" : 11 },
#     {"name" : "Bulus Rufus", "email" : "bulusrufus.gmail.com", "dept" : "Biology" , "level": 300, "ID" : 12},
#     {"name" : "Baba Love", "email" : "babalove.gmail.com", "dept" : "Computer Science" , "level": 400, "ID" : 13 },
#     {"name" : "Gana Lois", "email" : "ganalois.gmail.com", "dept" : "Chemical Engineering" , "level": 500,"ID" : 14},
#     {"name" : "Tobi Tolu", "email" : "tobitolu.gmail.com", "dept" : "Agric Exetension" , "level": 200 , "ID" : 15 },
#     {"name" : "Nma Mohd", "email" : "nmamohd.gmail.com", "dept" : "Biochemistry" , "level": 500 , "ID" : 16},
#     {"name" : "Ndagi Baba", "email" : "ndagibaba.gmail.com", "dept" : "Mathematics" , "level": 200 , "ID" : 17},
#     {"name" : "Lucas Dodo", "email" : "lucasdodo.gmail.com", "dept" : "Geography", "level": 400, "ID" : 18}
# ]

# # graduating list(aluminas)
# Alumina = [

# ]

# # a fuction to add more student 
# def add_one_student(new_student):
#     return  students.append(new_student)

# # #function to add multiple student
# # def add_multiple_student(new_students):
# #     return  students.extend(new_students)


# add_one_student({
#        "name" : "Bola Are",
#        "email" : "bolaare.gmail.com",
#        "dept" :  "Animal Science",
#         "level" : 200, 
#         "ID" : 19
#     })
# print(students)


# add_multiple_student([{
#        "name" : "Bola Are",
#        "email" : "bolaare.gmail.com",
#        "dept" :  "Animal Science",
#         "level" : 200, 
#         "ID" : 19
#     }, {
#        "name" : "Bola Tito",
#        "email" : "bolatito.gmail.com",
#        "dept" :  "Plant Science",
#         "level" : 300, 
#         "ID" : 20
#     }])
# print(len(students))

# # the function to  remove student using ID
# def  remove_student( ID):
#     for student in students:
#         if student["ID"] == ID:
#           return  students.remove(student)    	
   

# remove_student(1)
# print(students)


# # function to update studnts
# def update_student(ID, updateValues):
#     for student in students:
#         if student ["ID"] == ID:
#             student["name"] = updateValues.get("name", student["name"])
#             student["dept"] = updateValues.get("dept", student["dept"])
#             student["email"] = updateValues.get("email", student["email"])
#             student["level"] = updateValues.get("level", student["level"])

#             print(f" updated student ID{ID}: name: {student ['name']}, dept : {student['dept']}")
#             return 
        
# update_student( 1,{
#        "name" : "Bola Are",
#        "email" : "bolaare.gmail.com",
#        "dept" :  "Animal SciencSe",
#         "level" : 200, 
#     })


# print(students)


# #creating a function that will promote students to next level and movindg finalist to alumina list           

# def promte_student():
#     for student in students:
#        student['level'] += 100
#        if student ["level"] > 500:
#           Alumina.append(student)
#           students.remove(student)
           

# promte_student() 
# print(Alumina)
# print(students)


# #create afunction that around the students order or disorder

# def sort_students( method='ascending', sort_type='name'):
#     reverse = True if method == 'descending' else False
#     try:
#         sorted_students = sorted(students, key=lambda x: x[sort_type], reverse=reverse)
#         return sorted_students
#     except KeyError:
#         print(f"Invalid sort type: {sort_type}")
#         return []


# print(sort_students(method='descending', sort_type='level'))

#assigning a faculty to each student abovw
# def assign_faculty_to_students(students):
#     dept_to_faculty = {
#        "Mathematics" : "SPS",            
#         "Physics": "SPS",         
#         "Forestry": "AGRIC",        
#         "Biology": "SLS",       
#        "Computer Engineering": "SEET", 
#         "Micro Biology": "SLS",      
#         "statistic": "SPS",         
#        "Crop Production" : "AGRIC",      
#        "Mechanical Engineering" : "SEET",         
#       "Telecom Engineering": "SEET",
#        "Chemical Engineering" : "SEET" ,
#          "Agric Exetension": "AGRIC" ,
#         "Biochemistry" : "SLS", 
#           "Geography": "SPS",
#           "Computer Science" : "SICT"            
#     }
    
    
#     for student in students:
#         dept = student['dept']
#         faculty = dept_to_faculty.get(dept)
#         student['faculty'] = faculty

#     return students

# print(assign_faculty_to_students(students))

# #another assignment

# def grading_system(score):
#        if 70 <= score <= 100:
#             return "A"
#        elif 60 <= score <= 69:
#             return "B"
#        elif 50 <= score <= 59:
#             return "c"
#        elif 45 <= score <= 49:
#             return "D"
#        elif 40 <= score <= 44:
#             return "E"
#        elif 0 <= score <= 39:
#             return "F"
#        else:
#             return " no assesement"
# grades = grading_system(70)
# print(grades)

# def calculate_grades_and_remarks(students_scores):
#     # Define grading criteria
#     grading_criteria = [
#         (70, 100, 'A', 'Excellent'),
#         (60, 69, 'B', 'Very Good'),
#         (50, 59, 'C', 'Good'),
#         (45, 49, 'D', 'Fair'),
#         (40, 44, 'E', 'Poor'),
#         (0, 39, 'F', 'Fail')
#     ]
    
#     # Function to determine grade and remark
#     def get_grade_and_remark(total_score):
#         for min_score, max_score, grade, remark in grading_criteria:
#             if min_score <= total_score <= max_score:
#                 return grade, remark
    
#     # Calculate total scores, grades and remarks
#     results = {}
#     for student, subjects in students_scores.items():
#         results[student] = {}
#         for subject, scores in subjects.items():
#             total_score = scores['exam'] + scores['ca']
#             grade, remark = get_grade_and_remark(total_score)
#             results[student][subject] = {
#                 "total_score": total_score,
#                 "grade": grade,
#                 "remark": remark
#             }
    
#     return results

# # Example usage with ten students
# students_scores = {
#     "Student 1": {"Math": {"exam": 80, "ca": 15}, "English": {"exam": 65, "ca": 20}},
#     "Student 2": {"Math": {"exam": 50, "ca": 20}, "English": {"exam": 55, "ca": 15}},
#     "Student 3": {"Math": {"exam": 70, "ca": 10}, "English": {"exam": 60, "ca": 18}},
#     "Student 4": {"Math": {"exam": 30, "ca": 5}, "English": {"exam": 40, "ca": 10}},
#     "Student 5": {"Math": {"exam": 75, "ca": 15}, "English": {"exam": 68, "ca": 17}},
#     "Student 6": {"Math": {"exam": 45, "ca": 10}, "English": {"exam": 50, "ca": 10}},
#     "Student 7": {"Math": {"exam": 60, "ca": 10}, "English": {"exam": 70, "ca": 20}},
#     "Student 8": {"Math": {"exam": 85, "ca": 15}, "English": {"exam": 78, "ca": 22}},
#     "Student 9": {"Math": {"exam": 40, "ca": 10}, "English": {"exam": 45, "ca": 12}},
#     "Student 10": {"Math": {"exam": 55, "ca": 15}, "English": {"exam": 60, "ca": 20}}
# }

# results = calculate_grades_and_remarks(students_scores)

# for student, subjects in results.items():
#     print(f"Results for {student}:")
#     for subject, details in subjects.items():
#         print(f"  {subject}: Total Score = {details['total_score']}, Grade = {details['grade']}, Remark = {details['remark']}")
#     print()


