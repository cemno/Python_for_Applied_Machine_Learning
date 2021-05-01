"""
  This week we will cover the following tasks:
  1. strings and their assignment
  2. loops (for and while)
  3. creating functions
  4. file input and output
  5. saving pickles
"""

"""
  ######## 1. assigning strings
"""
# As you start to utilise python more and more you will realise that strings are an
# integral component. Including printing to screens, writing information to log files,
# file name, and many more.
# last week we assigned strings to variables:
# str0 = "Hello world!"
# Now we will play with strings on a deeper level.
# 1.1 adding strings together, the addition sign + has an interesting properties in
# python, we will see one now with strings.
# your task, create variables with strings where one of them is the space ' ' key.
# then add them together: str = str1 + str2 + str3
# Then print out the result.
str1 = "Hello"
str2 = " "
str3 = "World"
print(str1 + str2 + str3)
import numpy

# So in this case the + operator is a concatenation operator. This can come in very
# handy at different times!
# 1.1 What about if we want to include numbers in the string like str = "pi = 3.14159"
# but we want to assign them dynamically. Like flt = 3.14; str = "pi" + flt does that work?
flt = numpy.pi
str1 = "pi = " + repr(flt)
print(str1)

# so how do we fix this? Well we will explore two options:
# the % operator and the .format call.
# let's start with % basics
# "str %s %d %f" % ( 'string', 1212, 3.14 )
# but first let's create individual strings for a string, int, float input

str1 = "%s" % "string"
int1 = "%d" % 1212
float1 = "%f" % 3.14

# now like the example above let's create a single string with four inputs:
# 'string', 1212, 3.14159, 0.001

print(str1 + " " + int1 + " " + float1 + " " + "%f" % 0.001)

# Now what about if we want to do something a little more fancy with say the
# integer or floats such as trailing or leading zeros.
# Sometimes, say for file saving we want leading zeros to keep sorting
# true say 01, 02, 03, ..., 10 compared to 1, 10, 2, 3, ..., 9
# we do this by specifying the number of total values in an integer:
# str0 = '%010d'%(1212), in this case we will get a total of 10 values, or 6
# leading 0's for 1212. Try it for different totals.

print("%010d" % 1212)

# Can you work out how to do it with trailing zeros on a decimal point floating
# number? HINT use the decimal point.

print("%.4f" % 0.0100000)

# 1.2 how do we use the .format operator to do exactly the same thing?
# In the pdf you see one example, you should note that instead of the '%f' we can
# simply use '{}' to indicate the insertion point.
# Basics: let's replicate the pdf without any type assignments.

print("{} test {} {} {}" .format("string", 1212, 3.14, 0.001))

# but in that same format we can assign types which helps us be more specific
# about the type information (like when we added leading zeros).
# For this two things are important in the {} -> {index in the .format():type information}
# i.e. '{1:} {0:0.02f}'.format( 3.14159, 1212 )
# in this case we would output >>> '1212 3.14' note that the index has told it
# Try it yourself


# to look at a certain place. This is not necessary, and to be honest I have never
# found it overly helpful. Just put the values in .format() in the right order?
# Can you think of a case where it might help?


# Overall you can pick the version you find easier.


"""
  ######## 2. For and While loops
"""
# So in the pdf I gave a very brief example of a for and a while loop.
# Let's start with the while loop and show the benefits and the major issues.
# while <condition is true>:
#   <do something>
# So as long as the condition is true the loop will continue, this is both a
# benefit and a pitfall i.e. what happens if we never change the condition...
# Infinite loop.
# let's do a simple condition to show this behaviour while x > y
# let x make x = 10 and y = 0 and c = 1 as a strating point
# then within the while loop print out y then add c to the y.
# how long does the loop go for?


# now change c to 0.1 and see what happens? Don't forget to reset y!!


# now let's change c to -1 and see what happens? Don't forget to reset y!!
# You'll need to manually stop the code


# comment this out after you run it.

# So while loops are simple tools with some pitfalls.
# What about for loops? Firstly you need to give them a specific path to loop over.
# This can be anything from a list, dictionary, numpy array to a personally built path.
# Generally in the pdf I showed an example with a list. But here we will use an example
# that uses a new function "range". Range takes as input at least an upper limt, and
# returns a range of values. Let's play with that now, x = range( lower value, upper value )
# do this now for 0 to 10 then print it, what do you notice about the upper value?


# what happens if we envoke the list class


# however we don't need to do this for for loops. "range" has an iterator member that
# allows you to move over the values in the class 0->10 in this case.
# to simplify this, if we are starting at zero we can just do:
# x = range( upper limit ) try it now


# how does this help us with a for loop?
# well like I said we need something to iterate over, and a "range" is a perfect
# example of this.
# for iterator in list:
#   print( iterator )
# try this now using range instead of list.


# now let's look at some other things to do with a list.
# let's create a list of values [0,2,4,6,8,10,12,14,16,18,20]
# now let's create a for loop that iterates around this list using an iterator
# that depends on the "len" function.
# 'len' does exactly what you expect, get's the lenght of something.
# so len returns an integer, we can now use this in "range"
# for i in range( len( list ) )
# try this now with a print function


# essentially this is the same as:
# for i in x:
# but this gives us the addition of an iterator, this can come in handy at different
# times.
# Try doing this at home:


# Finally, for the basics let's consider dictionaries.
# d = {'str':'dog', 'pi':3.14159, 'int':1212}
# how do you think you would iterate through this? HINT remember keys()?


# we can also do this another way by envoking .items() this returns the key and
# the value.


# again different pieces of code call for different approaches. Just make sure
# you understand that there are usually multiple ways to do the same thing.
# this just shows you some of them.

"""
  ######## 3. Functions
"""
# Functions are way of storing methods of doing something that you might want to
# use repeatedly.
# Much like loops there is a standard structure to creating functions:
# def <function name>(input0, input1, ..., inputN):
# let's create a function called "rectangle_area" that calculates the area of
# a rectangle based on two inputs h and w.


# Along with having mutiple inputs you can also use multiple outputs.
# return out0, out1, ..., outM
# let's try this with the properties of a circle based on an input radius.
# first we need to define pi, it's up to you how many decimals you define it to.


# or alternatively we can use the inbuilt pi from the numpy library.
# Do you remember how to import numpy, try it now


# the internal numpy pi can be called simply as np.pi


# now let's create a function that takes as input, r(adius) and pi
# and outputs the diameter, the circumference, and the area.
# and then output each of them.


# There are lots of little tricks for functions, but two that can be quite handy
# are calling functions within another function and also having multiple inputs in one
# similar to what we did with print( *args )
# So, let's do both of those little tricks in one.
# Let's redefine circle_properties with multiple *r
# finish the header, but with one little trick. Let's create a default value for
# pi i.e. pi=<insert value>. Just remember that when creating default values they HAVE
# to come after non-default assigned variables. You cannot have
# def <function name>(var0, var1=3, var2, var3)
# Now let's complete this header:
# def circle_properties_multiple()
# now we need to create empty numpy.arrays to store the outputs from
# to do this we will use numpy.zeros( (rows, cols) ) where rows will be based on
# the length of r and we will make it a matrix so the cols will be 1.
# We want one of these variables for diameter, circumference, and area.
# finish off these assignments
# d = np.zeros(
# c = np.zeros(
# a = np.zeros(
# now we will iterate over r and use the circle properties function to create
# the values we need. Finish off this code
# for i in
# now based on the iterator we want to assign values to d, c, and a based on the r
# = circle_properties(
# now let's output this data
# return


# display the results.


# display the results per input... Try this in yor own time. USE A STRING!


# Obviously it would be just as simple in this case to pass the radius input
# as a list or numpy array. But this just shows you another way to do it.
# if you'd like try and do it as a list in your own time?

# def circle_properties_multiple( r, pi=np.pi ):

# alternatively we could pass the circle_properties function as an input like:
# def circle_properties_fn( *r, pi=np.pi, fn=None )
# ...
#     d[i], c[i], a[i] = fn( r[i], pi )
# print( circle_properties_fn( 3, 4, 5, fn=circle_properties ) )
# Try this in your own time.

# def circle_properties_fn( *r, pi=np.pi, fn=None ):


"""
  ######## 4. Text file input and output
"""
# Now we move onto saving text in a file. There are a number of uses for this and
# it's a good tool to have.
# Basically, we open a file, we write information to that file, and IMPORTANT we
# close the file. There is an easy way to do this, but for now we will do open and
# close as different functions to make it simple to remember.
# To open a file we need two pieces of information, the file name, and the openning
# properties like read (r), write (w), or append (a).
# fid = open( 'file.txt', 'w' ) # I'll set this to write as I do want to competely
# overwrite what is in that file.
# don't forget fid.close()! I always write both pieces of code straight away
# when I do this. Let's do it


# Look in the file you just created. It should be in your working directory
# Now how do we add stuff to a file?
# There are two options:
# fid.write( <string to write> ) does not automatically put a new line escape in
# print( <string to write>, file=fid  ) Remeber this does put a new line escape automatically
# Now create a new file (don't forget to close it!).
# create a string with the end line character '\n' at the end.
# inside this string insert a integer and float.
# use both the write and print to insert it into the file.
# Go.


# look in the file you just created.
# what happens if you switch the end of line characters?


# have a look at the two files and compare them.
# so depending on the output you are after these escapes can be very important.

# Now let's try appending something to the first file we put something in.
# remember you will use the 'a' instead of 'w'. Once you have done this, look into
# that file and see what is there?


# now what if we want to read in that information? Let's use the first file
# we created that has something in it, and that we appended too.
# In this case we will open the file with the 'r' read flag.
# first we will use the readlines function. I find this the easiest way to access
# text files. fid.readlines() reads in all the lines of in a file. Try to work out,
# using a for loop, how to display all the information to the screen.


# now a second way to do the same thing. Using readline. This way iteratively
# reads each line in the text file, compared to readlines which reads them all.
# now because we don't know the end condition we have to use a while loop.
# see if you can work it out.


# can anyone think of a problem with this method? what if l is empty deliberately?
# try it, manually put an empty line in the file.

"""
  ######## 5. Pickles: Saving other pythonic information
"""
# so this is how we store and read in text files, but what about lists, dictionaries,
# or even classes? In this case we would use the pickle library. It's a great tool
# for saving and loading information. The first thing we need to do is import pickle


# let's create a list and a dictionary. You decide what goes in them.


# now we will create a pickle and save them independently.
# We will do this in exactly the same way we would a text file from above.
# Except in this case we need to include the binary flag with our other flags.
# i.e. 'w' becomes 'wb', 'r' becomes 'rb' and so on.
# And instead of fid.write we have pickle.dump( information, fid ). Kind of like
# how we used the print function in the text file example. Also when creating a pickle
# the file should end in .pkl or something similar for standardisation.
# Create a file and pickle.dump the list, then create another one and pickle.dump
# the dictionary. Don't forget to close them!


# now how do we read this information back in?
# in this case we will use pickle.load( fid ). So obviously we need to open a file
# id here. Remember we need the binary flag as well. Try to load and print the information.


# but what if we want to save both the list and the dictionary?
# well you have multiple options, you can store them as a list, tuple, or dictionary.
# let's do it as a tuple. Save both the list and dictionary in one pickle.


# now let's try and load this?


# Obviously the key here is that you need to know what is in your pickle, this only works
# for tuples or lists. Try it in your own time.
