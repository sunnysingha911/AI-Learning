temperature = 15


if temperature > 30:
    print("Its warm")
    print("Drink Water")
elif temperature > 20:
    print("Its Nice")
else:
    print("Its cold")
print("Done")


age = 22

message = "Eligible" if age > 18 else "Not Eligible"

print(message)

high_income = True
good_credit = False

if high_income and good_credit:
    print("Eligible")
else:
    print("Not eligible")
