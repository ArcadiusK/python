x, k = map(int, input().split())
print(eval('{}=={}'.format(input(),k)))

n, x = map(int, input().split())
students = []
for i in range(x):
    students += [map(float, input().split())]
for ele in zip(*students):
    print(sum(ele)/x)

lines = [line.rstrip('\n') for line in file]

import datetime
for i in range(int(input())):
    print(abs(int((datetime.datetime.strptime(input(), '%a %d %b %Y %H:%M:%S %z') 
          - datetime.datetime.strptime(input(), '%a %d %b %Y %H:%M:%S %z')).total_seconds())))

print(a//b)
print(a%b)
print(divmod(a,b))
print(pow(a,b))
print(pow(a,b,m))
a, b, c, d = int(input()), int(input()), int(input()), int(input())
print(a**b + c**d)

from itertools import product
A = list(map(int, input().split()))
B = list(map(int, input().split()))
print(*list(product(A, B)))

from itertools import permutations
word, p = map(str, input().split())
for item in sorted(permutations(word, int(p))):
     print("".join(item))

from itertools import combinations
word, k = map(str, input().split())
for i in range(int(k)):
    for item in list(combinations(sorted(word),i+1)):
        print("".join(item))

from dateutil import tz          
from_zone = tz.gettz('UTC')
to_newyork_zone = tz.gettz('America/New_York')         
utc_time = datetime.datetime.strptime(datetime.now(), '%Y-%m-%dT%H:%M:%S.%fZ')
utc_time = utc_time.replace(tzinfo=from_zone)
eastern_time = utc_time.astimezone(to_newyork_zone)

from itertools import combinations_with_replacement
word, k = map(str, input().split())
for item in list(combinations_with_replacement(sorted(word), int(k))):
    print("".join(item))
