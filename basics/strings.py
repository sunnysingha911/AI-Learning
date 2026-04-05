course = "Python Programming"
message = """
Hi John

This is dev from Programming

"""
print(len(course))

print(course[-1])

print(course[0:3])

print(course[0:])
print(course[:3])

print(course[:])


first = "John"
last = "Doe"
full_name = f"{first} {last}"
print(full_name)


print(course.upper())
print(course.lower())
print(course.title())
print(course.strip())
print(course.find("Pro"))
print(course.replace("P", "j"))
print("Pro" in course)
print("Swift" not in course)
