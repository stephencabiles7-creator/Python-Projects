name = input("Enter a Word")
new_name=" "
for i in range (len(name)):
    if name[i] == "a" or name[i] == "A" or name[i] == "E" or name[i] == "e" or name[i] == "i" or name[i] == "I" or name[i] == "u" or name[i] == "U":
        continue
    new_name += name [i]
    

print(new_name)