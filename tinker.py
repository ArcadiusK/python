import re, sys
import numpy 


def conn(x_tensor, num_outputs, apply_activation=True):
     if(apply_activation):
        return tf.nn.relu(fc_layer)
    else:
        return fc_layer    
return conn(x_tensor, num_outputs, apply_activation=True)

my_dictionary = {}
for c in input():
    if c in my_dictionary: 
        my_dictionary[c] += 1
    else: 
        my_dictionary[c] = 1
sorted_d = sorted(my_dictionary.items(), key=lambda x: (-x[1], x[0]))
for t in sorted_d[0:3]:
    print(*t)
    
     

[print(one_line_out) for one_line_input in sys.stdin for one_line_out in re.findall('[\s:](#[a-f0-9]{6}|#[a-f0-9]{3})', one_line_input, re.IGNORECASE)]     


s, k = input().strip(), input().strip()
m = re.compile(k)
groups = m.search(s)
if groups:
    while groups:
        print("({0}, {1})".format(groups.start(), groups.end() - 1))
        groups = m.search(s, groups.start() + 1)
else:
    print((-1, -1))


from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):        
        print ('Start :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
    def handle_endtag(self, tag):
        print ('End   :',tag)
        
    def handle_startendtag(self, tag, attrs):
        print ('Empty :',tag)
        for ele in attrs:
            print ('->',ele[0],'>',ele[1])
            
MyParser = MyHTMLParser()
MyParser.feed(''.join([input().strip() for _ in range(int(input()))]))


import re
s = re.search(r'([a-zA-Z0-9])\1', input())
print(s.group(1) if s else -1)

m = re.search(r"\b(?=\w)" + re.escape(k) + "+"+r"\b(?!\w)", s)
print(m.group(1))

list_to_return = re.split("[.,]",input())     
if (re.match("^[A-Za-z0-9_-]*$", username):
print(bool(re.search(r"^[+-]?\d*\.\d+$",raw_input().strip())))
    
sorted(list(filter(lambda x: re.search(r'^[\w\d-]+@[A-Za-z0-9]+\.\w?\w?\w$',x),lst)))

    
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1      
    for child in elem:
        depth(child, level + 1)   
tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
    
#when node is xml
return(sum([len(i) for i in [_.attrib for _ in node.iter("*")] if i !={}]))
     
def person_lister(f):
    def inner(people):
        for one_person in sorted(people, key=lambda x: x[2]):
            print(f(one_person))
    return inner

utc_zone = tz.gettz('UTC')
nyc_zone = tz.gettz('America/New_York')
nyc_time = datetime.datetime.now()
utc_time = nyc_time.astimezone(utc_zone)
     
 per_day_list = [1, 2, 3, 4, 5]
 print('%s' % ' '.join(map(str, per_day_list)))     
     

a = []
for i in range(int(input())):
    a.append(list(map(float, input().split())))
print(numpy.linalg.det(a))

try:
    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)       
except:
    print("ARC Unexpected error:", sys.exc_info()[0])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)
    raise
     
print(time.strftime("%H:%M:%S:%f"))
     
from collections import deque
for i in range(int(input())):
    n, q = int(input()), deque(map(int, input().split()))
    for j in range(n - 1):
        if q[0] >= q[1]: q.popleft()
        elif q[-1] >= q[-2]: q.pop()
        else: break
    print("Yes" if len(q) == 1 else "No")


import itertools
k, m = map(int, input().split())
to_return = 0
all_lists = []
for i in range(k):
    all_lists.append(list(map(int, input().split()[1:])))
combinations = list(itertools.product(all_lists))
for one_combination in combinations:
    to_be_maximixed = sum([x**2 for x in one_combination])%m
    if to_return < to_be_maximixed:
        to_return = to_be_maximixed
print(to_return)

with open("case00-in.txt") as f:
    k, m = map(int, f.readline().split())
    sum = 0
    for i in range(k):
        l = list(map(int, f.readline().split()))
        max_ele = max(l[1:])
        sum += (max_ele**2)
        r = sum % m
    print(r)
f.closed

def start_repeating():
    global next_call
    next_call = next_call + 1
    threading.Timer( next_call - time.time(), start_repeating).start()

import itertools
mlist = []
for k, group in itertools.groupby(input()):
    mlist.append("(" + str(len(list(group)))+", " + str(k) + ")")
print(*mlist)

utc_time = datetime.datetime.strptime(parsed_json["trade_closed_at"], '%Y-%m-%dT%H:%M:%S.%fZ')
utc_time = utc_time.replace(tzinfo=from_zone)
eastern_time = utc_time.astimezone(to_newyork_zone)

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
