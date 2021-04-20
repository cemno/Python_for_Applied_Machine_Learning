"""
  Welcome to week 01 practical.

  We will the reinforcing what you learned in the lecture last week.

  Primarily:
  1. how to use comments
  2. print function
  3. creating variables.
  4. expressions and statements with arithmetic
  5. boolean operators
  6. and or operations
  7. the if statement
  8. basics of numpy
  9. lists, dictionaries, and tuples.
  10.mutable vs immutable
"""

###### 1. Commenting. ######

"""
  Commenting is an important part of good code. It should not be overdone, but it should
  also help a reader of your code understand what is going on.

  Using the three " is actually creating a string that can go over multiple lines, we
  will expand on that in a later tutorial but for now just know it's a nice way to highlight
  comments.

  The standard way to comment code is with the # operator. This is how I will primarily
  do it.

  Keep in mind that a lot of comments can sometimes make it more difficult to understand,
  it's a balancing act. From my experience it's usually better to have too much rather than
  not enough.
"""
# Throughout these templates I will try to comment as much as possible to ensure you
# can complete the tasks, however, if something is not clear I recommend that you
# add comments to the code that will help you understand in the future.

"""
  ###### 2. the print function ######
"""
# A)  Your first task is to print Hello World! again using either " or '

print("Hello world")

# B)  Can you separate the words and print them individually?
#     Below is the print function
#     print(*objects, sep=' ')
#       *objects  - any number of inputs to be printed.
#       sep       - how are we separating the inputs? Default is space.
#     There are others but for now let's concentrate on these two.
#     From this try and use multiple inputs to the print function i.e. print( 'hello', 'world!' ).
#     what is the result?

print('hello', 'world!')

#     Now let's try with a different separator, try whatever you like,
#     print( "hello", "world!", sep='+' )

print('hello', 'world!', sep = '+')


# C)  Can we print other things other than strings?
#     and can we combine different things?
#     print an integer and a float separately

#print() # int
#print() # float

#     now try and print a string, an integer, and a float in the same print function.

print('fünf', 5, 5.0) # string, int, float

"""
  ###### 3. Creating Variables ######
"""
#     How do we create variables?
#     Variables are used to store pieces of information such as strings, integers,
#     floats and a number of other things.
#     A variable name can contain upper or lower case letters, digits and the underscore _,
#     but, they cannot start with a number... If I can make one suggestion, try to make
#     your variable names easy to follow, i.e.
#     firstname = 'Michael'
#     surname = 'Halstead'
#     BUT! You have to be careful with what you call things, as certain names already
#     exist i.e. don't call something float = 'string' as float is an inbuilt python
#     object. You will start to know more of these as the semester continues, but
#     for now understand that it will cause problems if you overwrite something.
# A)  lets create a number of variables storing a string, an integer, and a float.
#     You can create any number of these, an example: str0 = 'Hello'

print( 'EXERCISE 3A')
str0 = "Hello"
a = 5
b = 5.0
print("Done.")
# B)  Now can you print these variables??

print( 'EXERCISE 3B' )
print(str0, a, b)

"""
  ###### 4. Arithmetic ######
"""
#     Python has some basic built in maths functions.
#     +   addition
#     -   subtraction
#     *   multiplication
#     /   division
#     **  power i.e. 2**3 = 8
#     //  floor division i.e. 3/2 = 1
# A)  Use each of these mathematical expressions and print the result

print( 'EXERCISE 4A')

print(a**b)

# B)  Try to use variables in these expressions.
#     i.e. x = 13
#          y = 3
#          z = x // y
#          print( z )

print( 'EXERCISE 4B')
x = 13
y = 3
z = x//y
print(z)


"""
  ###### 5. boolean operators ######
"""
# These operators are essentially True or False values that can be used for a number of things in python.
# One of the most common uses is if statements which we will cover later.
# A) Print the two types of boolean values (True or False) remember it is a capital letter at the start.
#    Create a variable of each value and print them too.

print( 'EXERCISE 5A')
t = True
f = False
print(t, "or", f)
# B) Creating from statements.
# you can also create booleans by using statements that would create a true or false situation.
# this can include bool_var = 5 > 4
# in this case if 5 is greater than 4 you would return True else it would be False.
# the following statements can be used:
# x==y  : x is equal to y
# x>y   : x is greater than y
# x<y   : x is less than y
# x>=y  : x is greater than or equal to y
# x<=y  : x is less than or equal to y
# x!=y  : x is not equal to y
# create two variables, x and y and assign them with numerical values.
# Using the above statements to output True or False values based on these variables.

print( 'EXERCISE 5B')
print(x==y)

"""
  ###### 6. and or operations ######
"""
# An extension of exercise 5 are the 'and' and 'or' operators.
# you can create a single boolean variable from a number of boolean operators.
# left and right   : both the left and the right side must return True else this outputs a False.
# left or right    : either the left or the right must be True for the function to output a True.
# A) create three variables (x, y, and z) and assign them numerical values.
#    use the operators from 5B to create True or False values based on these created variables using "and" or "or".
#    print the output.

print( 'EXERCISE 6A' )
x = 10
y = 5
z = 2
print("x == z*y and x/z == y: ", x == z*y and x/z == y+1)
print("x == z*y and x/z == y: ", x == z*y and x/z == y)

# B) the 'not' operator: essentially converts a False to a True
# x and not y means that if y is False then it becomes True
# not x and y means that if x is False it becomes True
# Use the variables from 6A and create a variable using the not operator

print( 'EXERCISE 6B' )
print("x == z*y and not x/z == y: ", x == z*y and not x/z == y)
print("x == z*y and not x/z == y: ", x == z*y and not x/z != y)
"""
  ###### 7. the if statement ######
"""
# If statements are a very handy programming tool that compares a boolean operator and if it's true will inact some code.
# the general idea is:
# if <something is True>:
#   <do something>
# elif <something else is True but not the thing on the "if" line>:
#   <do something else>
# elif .... (can have any number of these)
# else:
#   <catch all statement to do something>
# In the first line we try to get a True result from an operator i.e. x > y
# if this isn't true we try the elif command i.e. x == y
# Finally if none of the previous statements are true we go into the "else"
# Three really important things to keep in mind.
#  1. You must end the control line with ":" if you don't you will get errors. Try it below.
#  2. Indentation is very important!! After the control statement you MUST tab or space in! If you don't
#     you will get errors, try it below. Keep in mind that pycharm should automatically does this unless you changed something.
#  3. You can't leave the inside of an if statement empty. i.e. where I have <do something> you must do something!
# A) create a basic if statement using your variables from Exercise 6, just do a single compare function and just the if
#    (i.e. don't worry about elif or else):

print( 'EXERCISE 7A' )
if x == y * z:
    print(True)

# B) Now use the same control statement and create an elif and else to go with it.
# play around with this so it goes into the different parts of the if statement.

print( 'EXERCISE 7B' )
import random
schroedinger = random.randint(0,2)
if schroedinger == 1:
    print(True)
elif schroedinger == 0:
    print(False)
else:
    print("None")

# C) Now let's use multiple control statements in a single if (i.e. if x<y and x>z etc.)

print( 'EXERCISE 7C' )
if x == z*y and x/z == y:
    print(True)

"""
  ###### 8 lists, dictionaries, and tuples ######
"""
# lists, dictionaries, and tuples are storage classes that allow you to store all different types of information
# in an array like fashion.
# They are very very handy objects that we will use a lot in this course, particularly lists!
# In python please keep in mind that we start storing information at index 0. I will explain later why this is
# so important but it's good to keep in mind that the first element of a list and tuple is not index = 1 it is index = 0

###### 8A basics of a list
# lists are literally what they sound like, a basic object that can store lists of information. This can be any
# information and any combinations of information.
# 8A-1 Create two empty lists, you can do this with the l=[] brakets or by invoking l=list()

# empty list
print( 'EXERCISE 8A-1' )
note = list()
print("Done.")
# 8A-2 lists with something in them... use the [] brackets and create 2 lists with at least three elements in them.
# l = [1,2,3]
print( 'EXERCISE 8A-2' )
note = [3]
note.insert(0, 1)
note.append(2)
print(note)
# A handy method with lists  is the ability to create a list with N of the same elements
# x = [3]*3 try it now with either strings, floats or ints
note = note * 3
print(note)
# 8A-3 lists also come with some inbuilt methods that do various things.
# lst[0] gets the 0th element, keeping in mind that this is the first element.
#       we can substitute 0 for any index that exists within the list.
# lst.append( 3.14 ) will add 3.14 to the end of the current list! This is very handy if you want to add stuff to a
# list at the end.
# lst.insert( index, 3.14 ) will insert 3.14 at the specified index, remember this index has to be inside the existing list.
# lst.pop( index ) will remove the value at index from the list, once again index must be within the current list.
# try using each of these tools.

print( 'EXERCISE 8A-3' )

note.insert(0, 1)
note.append(2)
note.sort()
print(note)

####### 8B basics of a dictionary
# Dictionaries are another inbuilt storage class. However, instead of using an index in the same way that lists do
# dictionaries must have a key:value pair, where the key is how you index.
# 8B-1 You can create empty dictionaries in the same way that you would lists but instead of the [] you use the {} and
# the dict(). Try it now.

print( 'EXERCISE 8B-1' )
dit = dict()
dit1 = {}
print(dit == dit1)


# 8B-2 the interesting thing about dictionaries is that the key can be a string or a number.
# {'first':1, 2:2, 'three':'dog'} you can see in this example that I have used either a string or a number.
# Pay close attention to how I assigned the key and value pairs, there is a : between the two!
# You try this, create a dictionary with at least three key:value pairs.

print( 'EXERCISE 8B-2')
dit = {1:"cool",2:"notcool"}
print(dit)

# 8B-3 In much the same way as lists we can alter dictionaries.
# dit[key] where key is one of the keys in the dictionary will return that value in the key:value pairs.
# dit[key] = value will change the value at key to the new value. Or if the dictionary exists it will add a new key:value
#                  pair
# dit.pop( key ) will remove the key:value pair assigned to key from the dictionary
# dit.update( {key0:value0, key1:value1} ) will update the dictionary with the new information or add elements if they
#                                           aren't already there.
# Try each of these in turn.

print( 'EXERCISE 8B-3' )
print(dit[1])
print("Pop:", dit.pop(2))
dit[1] = "new"
print(dit[1])
dit[1] = "new"
print(dit)

###### 8C basics of a tuple # immutable
# Tuples are the final inbuilt storage object we will be teaching you here.
# they are very similar to lists apart from one very key difference. Once they are created you can not easily alter them.
# This is a very handy property of tuples at different times so keep it in mind.
# 8C-1 Create an empty tuple. (This isn't overly practical but it's good to know)
# like lists and dictionaries you can create emtpy objects with list() or ()
# You should notice that we have used all sorts of brackets now.
# Create empty tuples

print( 'EXERCISE 8c-1' )
emptytuple = tuple() # or ()
print(type(emptytuple))
# 8C-2 Now let's create tuples.
# First thing to keep in mind if you want a single item tuple you need to insert a comma after the first variable.
# If you don't do this it will just return a string, float, integer, etc. Try it now.

print( 'EXERCISE 8C-2' )
tpl = (2)
print(type(tpl))
# 8C-3 Now let's create a tuple with at least 3 elements. Again you can put whatever you want in there.

print( 'EXERCISE 8C-3' )
tpl = (1,2,3)
print(tpl)

# 8C-4 Finally, like previously mentioned you can not directly manipulate a list. Try it using the same ways that we tried
# with lists and dictionaries?
# If you want to manipulate a tuple you need to convert it to a list first
# lst = list( tup )
# lst**do manipulations
# tup = tuple( lst )
# Obviously not very convenient and if you plan on manipulating a tuple maybe it's better to use a list?
# the only thing that still works is accessing a tuple index (just can't change it):


print( 'EXERCISE 8C-4' )
print(tpl.index(3))
lst = list(tpl)
print(tpl)
print(lst)
lst.pop(1)
print(tuple(lst))

"""
  ###### 9 basics of numpy ######
"""
# numpy is a library that we included in our environment. It stands for numerical python.
# it allows us to work with vectors and matrices and do a number of things.
# The first thing we must learn when using external libraries (those that aren't standard in python) is that we have
# to import them. In this case we are going to import the numpy library.
# You can do this in two ways, it really depends on you:
# import numpy # this will import all libraries and when you call numpy you must use the full library name i.e. numpy.array()
# import numpy as np # is a standard nickname for numpy, the difference is that instead of using the full name (numpy) you
# only have to use np i.e. np.array().
# This choice is completely up to you. For this course we will be using "import numpy as np"
# Usually you would put this at the top of the file, but for now it's okay to do it here.
# Either way let's import it now.



# 9A Creating vectors and matrices from a list
# The first way that we will learn to create a matrix array is from a list.
# create a vector of length N
# x = np.array( [1,2,3, ..., N] ) # here is a list inside a np.array function call which converts to a vector.
# what you should notice is that the list inside the function contains only 1 dimension.
# now create a vector using the above command and print the shape of the vector to the screen x.shape
# keep in mind that the list should contain numbers in this case...

print( 'EXERCISE 9A' )
import numpy
def random_vec(n, value_range):
    lst = [random.randrange(1, value_range, 1) for i in range(n)]
    vec = numpy.array(lst)
    return vec
print(random_vec(100,100))

# now we will create a matrix
# So the above only had a single dimension, but sometimes we want to work with matrices.
# i.e both a row and a column dimension. We can do that using a list by simply adding the second
# dimension as a set of [] brackets

# x = np.array( [[1,2,3]] ) print out the shape too?
print(numpy.array([random_vec(100,100)]))

# what about if I want the dimensionality switched? How do we do that? [[],...,[]]
vec = numpy.array([random_vec(100,100),random_vec(100,100),random_vec(100,100),random_vec(100,100)])
print("Four random vectors as array: \n", vec)

print("vec * vec: \n", random_vec(100,100) * random_vec(100,100))

# you can also do this with the x.T operator (transpose)
print("Transposed:\n", vec.transpose())


# what if we want a 3x3 dimension matrix?
# see if you can work that out using the above formation?
mat = numpy.array([[1,2,3]]*3)
print(mat)
# And can we manipulate these arrays?
# we can access these in the same way we would access a normal matrix: x[row,col]
# we can display the values and print the values. Try this with all three of your np arrays, keep in mind the first
# one was a vector...
mat[1,2] = 5
print(mat)
# what happens if you created a matrix/vector of integers but now you want to insert a floating point number? Try it
mat[1,2] = 5.125
print("No change after inserting a float:\n",mat)

# how do we fix that?
mat = numpy.array([[1.,2,3]]*3)
mat[1,2] = 5.125
print("Create new Matrix:\n", mat)

# or alternatively change the type?
mat = mat.astype(float)
mat[1,2] = 5.125
print("Or better, change type:\n", mat)


## 9B mathematical operations.
# Numpy works in much the same way as normal mathematical operations but in this case you can do it on vectors and
# matrices.
# try to perform basic value math on what you created.

print( 'EXERCISE 9b')
mat = mat * 5
print("Matrix * 5:\n", mat)
print("mat / 2 + 12:\n", mat / 2 + 12)

## 9C matrix operations.
#  numpy can also work with matrices and vectors for vector operations.
# things like multiplication, and dot product are there, along with sum, mean, max, min operators.
# first let's use multiplication on the previous matrices, remember dimensions have to hold so if
# you need to create new ones do it now.

print( 'EXERCISE 9C' )
mat0 = (mat + 2.5)**2 * 1.43
mat1 = (mat / 0.2 + 2) * 6
print("+\n", mat0+mat1)
print("-\n", mat0-mat1)
print("*\n", mat0*mat1)
print("/\n", mat0/mat1)


# We can also perform row or column wise operations.
# In this case x[3,3] and y[1,3]or[3,1] what happens if we perform mathematical operations on x using y?
# Try an array of both [1,3] and [3,1] what happens? What do you see?
mat0 = numpy.array([[1,2,3]])
mat1 = numpy.array([[1],[2],[3]])
mat2 = mat0 * mat1
print(mat2)

# You can do this with all the maths operations. This is handy knowledge to have in your toolkit for later.
# now let's look at the dot product
# mo (1x3) m2 (3x3) output should be 1x3
# d = np.dot( mat0, mat1 )
vec = [1,2,3]
vec1 = [3,2,1]
print("Dot product (1x3 * 1x3):\n", numpy.dot(vec, vec))
print("Dot product(3x3 * 1x3):\n", numpy.dot(mat2, vec))
print("Cross product:\n", numpy.cross(vec,vec1))


# there are a number of other functions available. You can find these by researching the numpy site or just using google
# in general. Here are a couple more:
# x = mat.mean() # mean of the matrix
# x = mat.sum() # the summation of the matrix
# x = mat.min() # the minimum value in the matrix
# x = mat.max() # the maximum value in the matrix.
# Try them out now.
print("Mean: ", mat2.mean())
print("Sum: ", mat2.sum())
print("Min: ", mat2.min())
print("Max: ", mat2.max())

# Any Questions?
