for number in range(1, 10, 2):
    print("Attempt", number, (number) * ".")


print(type(range(5)))


number = 100
while number > 0:
    print(number)
    number //= 2


command = ""

# while command.lower() != "quit":
#     command = input(">>")
#     print("Echo", command)

print("-------------------")

count = 0
for number in range(1, 10):
    if number % 2 == 0:
        count += 1
        print(number)
print(f"No of even no is {count}")
