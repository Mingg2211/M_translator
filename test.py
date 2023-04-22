import string

tmp = 'minggz'

if tmp[-1] not in string.punctuation :
    tmp = tmp + '.'
print(tmp)