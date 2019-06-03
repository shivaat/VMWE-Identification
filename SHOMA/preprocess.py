import numpy as np

def wordShape(word):
    vector = np.zeros(7)
    # 1 initial captial
    # 2 all char capital
    # 3 has #
    # 4 has @
    # 5 is URL
    # 6 is number
    # 7 has number
    if word[0].isupper():
        vector[0] = 1
    if word.isupper():
        vector[1] = 1
    if word[0] == '#':
        vector[2] = 1
    if word[0] == "@":
        vector[3] = 1
    if "http" in word[0:6]:
        vector[4] = 1
    if word.isdigit():
        vector[5] = 1
    if any(char.isdigit() for char in word):
        vector[6] = 1
    return vector

# TEST        
#print(wordShape('WORD26'))
