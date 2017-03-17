def conn(x_tensor, num_outputs, apply_activation=True):
     if(apply_activation):
        return tf.nn.relu(fc_layer)
    else:
        return fc_layer    
return conn(x_tensor, num_outputs, apply_activation=False)

def find_needle(haystack): return 'found the needle at position %d' % haystack.index('needle')

import collections
mdict = collections.OrderedDict()
counter_of_disctinct = 0
for i in range(int(input())):
    word = input()
    if word in mdict:
        mdict[word] +=1
    else:
        mdict[word] = 1
        counter_of_distinct += 1
print(counter_of_distinct)
print(*mdict.values())
    

import numpy
A, B = numpy.array(list(map(int, input().split()))), numpy.array(list(map(int, input().split())))
print(numpy.inner(A, B))
print(numpy.outer(A, B))

print(str(round(math.degrees(math.atan(float(input())/float(input())))))+'°')
    
print(sum([(ele in A) - (ele in B) for ele in ta]))     
     
ta, A, B = [input().split() for _ in range(4)][1:]
A, B = set(A), set(B)
happiness = 0
for ele in ta:
    if ele in A:
        happiness += 1
    elif ele in B:
        happiness -= 1
print(happiness)
     
import numpy
n, m = map(int, input().split())
my_array = []
for i in range(n):
    my_array.append(list(map(int, input().split())))
print(numpy.prod(numpy.sum(my_array, axis = 0)))
print(numpy.max(numpy.min(my_array, axis = 1)))

print(*sorted(input(), key=lambda x: 'z'+ str(not (int(x) % 2)) + x if x.isdigit() else x.swapcase()), sep='')

n, m = map(int, input().split())
elements = []
for i in range(n):
    elements.append(list(map(int, input().split())))
column = int(input())
for line in sorted(elements, key = lambda x: x[column]):
    print(*line)

response = requests.put()
    
def product(fracs):
    t = reduce(lambda x,y : x*y, fracs)
    return t.numerator, t.denominator

cube = lambda x: x**3

def fibonacci(n):
    toreturn = []
    a, b = 0, 1
    if n == 1:
        toreturn.append(0)
    elif n > 1:    
        toreturn.append(0)
        toreturn.append(1)
        for i in range(n-2):         
            a, b = b, a + b    
            toreturn.append(b)
    return toreturn


n = input();
s = map(int, input().split())
print(any(map(lambda x: str(x) == str(x)[::-1], s)) and all(map(lambda x: x>0, s)))

n = int(input())
iss = list(map(int, input().split()))
s= ""
p=""
for i in range(n):
    if iss[i] > 0:
        s = "good"
    else:
        s = bad
        break
if s == "good":
    for i in range(n):
        if str(iss[i]) == str(iss[i])[::-1]:
            p = "good"
            break
        else:
            p = "bad"
print(s == "good" and p == "good")

eval(input())

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
