#1
import math
from math import expm1

name = [c for c in "ANH PHA"]
print(name)

#2
print([x for x in range(1,10,2)])

#3a
sum1 = 0
for x in range(1,10,2):
    sum1 += x
print(sum1)

#3b
sum2 = 0
for x in range(1,7):
    sum2 += x
print(sum2)

#4
mydict={'a': 1,'b':2,'c':3,'d':4}
print([x for x in mydict.keys()])
print([x for x in mydict.values()])
print([(x, mydict[x]) for x in mydict])

#5
courses=[131,141,142,212]
names=['Maths','Physics','Chem', 'Bio']
for x in range(4):
    print((courses[x],names[x]))

#6
not_consonants = ['u', 'e', 'o', 'a', 'i']
s = 'jabbawocky'
cnt = 0
for x in s :
    if x not in not_consonants:
        cnt += 1
print(cnt)

cnt = 0
for x in s :
    if x in not_consonants:
        continue
    cnt += 1
print(cnt)

#7
for a in range (-2, 4):
    try:
        print(10/a)
    except:
        print('can’t divided by zero')

#8
ages=[23,10,80]
names=['Hoa','Lam','Nam']
list_tuple = [(ages[x], names[x]) for x in range(3)]
print(sorted(list_tuple, key= lambda x : x[0]))

#9
file = open("firstname.txt", "r")
print(file.read())
file = open("firstname.txt", "r")
for x in file:
    print(x)

## DEFINE A FUNCTION
#1
def get_sum(x, y):
    return x+y
print(get_sum(3,4))

#2
import numpy as np
from numpy.linalg import matrix_rank
M = np.array([[1,2,3],[4,5,6],[7,8,9]])
V = np.array([1,2,3])
print(matrix_rank(M), M.shape)
print(matrix_rank(V), V.shape)

#3
M3 = np.array([[col*3+row+3 for row in range(3)] for col in range(3)])
print(M3)

#4
transposed_M = np.array([[row[j] for row in M] for j in range(len(M[0]))])
print(transposed_M)
transposed_V = np.array([[V[i]] for i in range(len(V))])
print(transposed_V)

#5
from numpy.linalg import norm
x = np.array([2,7])
norm_x = norm(x)
print(norm_x)
normalized_x = x / norm_x
print(normalized_x)

#6
a=np.array([10,15])
b=np.array([8,2])
c=np.array([1,2,3])
print(a+b)
print(a-b)
print("cannot do a-c, shapes (2,) (3,)")

#7
print(a*b)

#8
A=np.array([[2,4,9],[3,6,7]])
print(matrix_rank(A), A.shape)
print(A[1][2])
print([col[1] for col in A])

#9
rand_matrix = np.random.randint(-10,10,size=(3,3))
print(rand_matrix)

#10
print(np.eye(3))

#11
print(rand_matrix.trace())
trace_loop=0
for x in range(len(rand_matrix)):
    trace_loop += rand_matrix[x][x]
print(trace_loop)

#12
a = np.array([1,2,3])
d = np.diag(a)
print(d)

#13
A = np.array([[1,1,2],[2,4,-3],[3,6,-5]])
determinant = np.linalg.det(A)
print(determinant)

#14
a1=[1,-2,-5]
a2=[2,5,6]
m = np.array([[x1,x2] for x1,x2 in zip(a1,a2)])
print(m)

#15
import matplotlib.pyplot as plt
x = np.linspace(-5, 6)
plt.plot(x, x**2)
plt.show()

#16
x = np.linspace(0,32, 5)
print(x)

#17
x = np.linspace(-5,5, 51)
plt.plot(x, x**2)
plt.show()

#18
x = np.linspace(-5,5, 10)
plt.plot(x, np.exp(x))
plt.title('exp(x)')
plt.xlabel("x")
plt.show()

#19
x = np.linspace(0.01,5)
plt.plot(x, np.log(x))
plt.show()
