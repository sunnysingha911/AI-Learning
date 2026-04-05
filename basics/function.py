def greet(first_name, last_name):
    print(f"Hi {first_name} {last_name}")
    print("Welcome aboard")


def get_greeting(name):
    return f"Hi {name}"


greet("John", "Doe")

print(get_greeting("John"))


def increment(number, by=1):
    return number + by


print(increment(2, 5))


def multiply(*numbers):
    for number in numbers:
        print(number)


multiply(2, 3, 4, 5, 6)
