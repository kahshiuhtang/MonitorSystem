import random
import string


def get_random_string(length):

    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters)
                         for i in range(length))
    # print random string
    return result_str


def get_score(s):
    tot = 0
    for idx in range(len(s)):
        tot += pow(7, idx+1) * ord(s[idx])*120
    return tot


s = set()
words = set()
for i in range(100000):
    st = get_random_string(int(random.random()*50 + 1))
    score = get_score(st)
    if score in s and st not in words:
        print(i)
        print("Error")
        break
    s.add(score)
    words.add(st)
