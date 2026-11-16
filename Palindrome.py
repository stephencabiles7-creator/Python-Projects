origWord = input("Enter a Sample word: ")
new_word =""    
for i in range (len(origWord)-1,-1,-1):
    new_word += origWord[i]
if origWord == new_word:
    print("The word is a palindrome")
else:
    print("The word is not a palindrome")
    
print(new_word)
