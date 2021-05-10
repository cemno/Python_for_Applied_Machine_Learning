"""
  week 3 topics
  1. classes
  2. data visualisation
  3. image manipulation
"""
import matplotlib.pyplot as plt

"""
  ####### Import area
"""
import numpy  # we always seem to use this so I will import it for you.
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import skimage.color as skcol

"""
  ####### Preamble
"""
# here are the flags we will use to run this.
# In future you will create these. See below how they are used. This is just to
# benefit us in reading the outputs. Previously we have had all the exercises
# printed to screen. Now we can choose which ones we print.
runclass = False
runvisual1 = False
runvisual2 = False
runimage = True

"""
  ####### 1. Classes
"""
# are we running the class information?
if runclass:
    # Classes are a great way to group common methods (functions) and members (variables) together in one space.
    # In last weeks practical we created a function that calculated the diameter, circumference,
    # and the area. Now we will do exactly the same thing in a class.
    # The inbuilt function __init__ is something that we will use throughout our classes.
    # When you initialise the class you can insert information that get stored.
    # First let's create the class called circle_props
    class circle_props:

        # The next step is to assign any shared variables.
        # These are static variables that are assigned to all circle_props classes
        # pi = np.pi # this creates a member self.pi equal to np.pi
        # We could do this here but I prefer to pass them in a different way.
        # Instead we will use an inbuilt class method __init__( self, )
        # __init__ does exactly what you would expect, creates the ability to initialise a class
        # based on input variables.
        # NOTE: For all methods of a class the first variable needs to be self, this ensures
        # the "function" is assigned to the class, giving it access to all class members and methods.
        # Let's create the __init__ method with a radius and pi input.
        def __init__(self, radius, pi_input=numpy.pi):
            self.pi = pi_input
            self.radius = radius

        # to assign an input variable to the class we need to use self.<variable name> = inputn
        # try to assign radius to self.r and pi to self.pi

        # so now self.r and self.pi are members in the class that we can use in any of our functions.
        # Okay, so let's try it. Create a method that calculates the diameter based on self.r
        # Don't forget self as the first variable
        def area(self):
            area = self.pi * (self.radius ** 2)
            return area

        def circumference(self):
            circumference = 2 * self.pi * self.radius
            return circumference

        def diameter(self):
            diameter = 2 * self.radius
            return diameter

        # Now create a method for the other two, you'll need self.pi
        # aggregation method
        # ?

        # the update method
        def update(self, radius=None, pi_input=None):
            if radius:
                self.radius = radius
            if pi_input:
                self.pi = pi_input

        def print_circle(self):
            print("My circle: ")
            print("radius: " + repr(self.radius))
            print("area: " + repr(self.area()))
            print("circumference: " + repr(self.circumference()))
            print("diameter: " + repr(self.diameter()))


    # okay let's calculate these statistics?
    # first we create the class, remember the parameters that were in __init__.
    circle1 = circle_props(5)
    circle1.print_circle()
    circle1.update(10)
    circle1.print_circle()

# output the statistics individually
# okay so we can now output the statistics individually, what about if we want to do it
# all in one go.
# Go back to your class and insert a method calc_all( self, ) that uses the other three
# functions to output in a single line. In this case, in exactly the same way we call
# members we can call methods. self.<methodname>()
# Then use that new method and output the results.


# Now, what if we want to update the radius? In this case we'd have to create a whole new
# class. Instead, let's create an update() function. That takes a new radius as input
# and changes self.r... The run it all again and update then output the results


# This is the basics of how to build up a class. Over the course of these practicals
# I will slowly introduce more information to you that will improve the way you build them.

"""
  ####### 2. data visualisation
"""
# Let's do some plotting with matplotlib
# Plotting is a great tool for visualising data. I have set up five exercises here
# but we will only do three of them. I really encourage you to try the others on your
# own. The solutions will be provided for you.
# The first thing you need to do is add import matplotlib.pyplot as plt in your import area.
# Are we running the visual? Do you need to change the flags?
if runvisual1:  # UNCOMMENT THIS LINE!!!
    pass
    # Okay so let's do some basic plots
    # So what we need is data on an x axis and data on a y axis: x and f(x).
    # In this case x will be an equally spaced vector (numpy). We will use the np.linspace( start, finsish, number of steps )
    # function to do this. Let's try and create data starting at 0, ending at 6 pi's, with 100 steps.
    steps = 6
    x = numpy.linspace(0, 6 * numpy.pi, steps)  # change step size


    # great, now what about f(x). Well numpy also has functions like sin and cos, let's do np.sin( x ) for our f(x)
    def f_sin(x):
        return numpy.sin(x)
    fx = numpy.sin(x)

    # now the interesting stuff. Let's create a figure using plt.figure()
    # input the x label using plt.xlabel( <insert name> ) and f(x) label with plt.ylabel( <insert name> )
    plt.figure()
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    #plt.subplot(211)
    # finally let's plot the data with the basic plt.plot() function... Can you work it out?
    plt.plot(x, f_sin(x), 'r--', label = "sin(x)") # or plt.plot(x, fx, 'r--')
    # and then we need to show it
    #  plt.show()
    # What about if we want to plot multiple things on the same plot?
    # Can you work it out? Try it in your own time.
    # We already have x and fx which we will now change to sinx cosx
    #plt.subplot(212)
    #plt.plot(x, numpy.sin(x))
    plt.plot(x, numpy.cos(x),  'g:', label = "cos(x)")
    plt.legend(loc = 0) #0 equals automatically best position
    plt.title("Sin(x) and Cos(x) at a step size of " + repr(steps))
    plt.show()
    plt.close()
    # Now let's create a plot using subplots. So let's say 4 plots in a 2*2 matrix.
    # We will again use the linspace for x that we created earlier.
    # create sin(x) cos(x) sin(x**2) sin(x)/4
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(x, numpy.sin(x), 'r')
    plt.title("f(x) = sin(x)")
    plt.subplot(222)
    plt.plot(x, numpy.cos(x), 'b')
    plt.title("f(x) = cos(x)")
    plt.subplot(223)
    plt.plot(x, numpy.sin(x ** 2), 'r:')
    plt.title("f(x) = sin(x**2)")
    plt.subplot(224)
    plt.plot(x, numpy.sin(x) / 4, 'r--')
    plt.title("f(x) = sin(x)/4")
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.95, hspace=0.5,
                        wspace=0.30)
    fig.suptitle("functions")

    plt.show()
    plt.close()
    # Now we will use the plt.subplots( rows, cols ) where we have a matrix of plots.
    # but in this case we return a fig and an axis from plt.subplots(). We have 4 plots so we'll do it as a 2*2 matrix.


    # Creating the title on the 'fig' using fig.suptitle not very intuitive but basically a super title.


    # Now let's plot on the first axis, [0,0] in the same way that we plt.plot but in this case ax[row,col].plot()


    # set the axis titles in this case we need to use ax[row,col].set( xlabel=<string>, ylabel=<string> )


    # now let's set the title... it's similar again but we use .set_title( <string> )


    # What about the other 3 plots... but after the x and y in plot(x,y) we'll change the colours.
    # Example plot( x, y, 'r' ) will plot red
    # plot( x, y, '--r' ) will plot dashed lines in red
    # plot( x, y, ':g' ) will plot dotted green lines. You could also use b for blue...


    # now let's show the plot


    # now let's run the script again but increase the number of steps in linspace. Say 2000, and what do you see?
'''
The step size is important for the "accuracy" of the plots, especially fpr the sin function with the power.
'''
if runvisual2:
    # Now we are going to do some scatter plotting, another very handy plotting tool for data visualisation.
    # But first we need to create some data, and in this case we will create normally distributed data along the x and y axis.
    # to do this we need a mean location as a numpy array: np.array( (mu_x, mu_y) )
    # and a sigma (covariance): np.array( [[x0, x1][y0, y1]] ). For more information on mean and covariance for the normal distribution
    # please see: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    # let's create the numpy arrays for the mean and covariances, dist0 = mu[3,2] cov[[1, 0.5][0.5,1]]
    #                                                             dist1 = mu[-1,1] cov[[2,0][0,2]]
    # mu_x = 100
    # mu_y = 50
    # mu = numpy.array((mu_x, mu_y))
    # sigma = numpy.array([[x0, x1][y0,y1]])

    # Okay now to create some normally randomly distributed points.
    # For this we need np.random.multivariate_normal( mean, cov, number of points )
    # Try to do that now based on what we just created, remember we need two distributions. And we'll create 100 points.
    X0 = numpy.random.multivariate_normal([3,2], [[1,0.5], [0.5,1]], 100)
    X1 = numpy.random.multivariate_normal([-1,1], [[2,0], [0,2]], 100)
    # okay now we have two distributions, now we just need to plot them.
    # let's look at the data itself using the .shape function, plot the shape of both distributions.
    print(X0.shape); print(numpy.shape(X0))
    print(X1.shape); print(numpy.shape(X1))

    # In this case we manually create a figure, plt.figure()
    plt.figure()

    # Then we need to do a scatter plot in the same way we did plot in the first example.
    # Where scatter( <x data of distribution N>. <y data of the distribution N>, c=<colour string like 'red'>, label=<legend name> )
    plt.scatter(X0[:, 0], X0[:, 1], c='red', label='D0')
    plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='D1')


    # now label the axis and give it a title plt.xlabel, ylabel, title
    plt.xlabel('X')
    plt.xlabel('Y')
    plt.title('Two multivariate distributions')

    # now show the figure
    # plt.show()

    # but this time we will also use plt.savefig( <savename.pdf> ) to save a version of the figure.
    plt.savefig("data/week04/fig_multivariate_dist.png")

    # view the image, it should be in your working directory.


    # Finally in your own time try to plot a histogram...
    # plotting a histogram and the distributions
    # this is just a 1d plot so only 1 mean and deviation (sigma) are required
    # In the previous example we did multivariate, now it's single...


"""
  ####### 3. image manipulation
"""
if runimage:
    pass
    # Now we will play with some images. As part of this weeks data you should have also downloaded two images.
    # one of these images is an RGB image, the other is a black and white image.
    # In this case we call the RGB image the input and the other a mask or label image.
    # These are two very important parts of machine learning, the RGB image is fed to a model, and we try to get it to
    # predict the mask...
    # But let's just play with the images for now. First let's import imsave and imload from skimage.io.
    # In the import section you will do: from skimage.io import imsave, imread
    # This does exactly what is said in the line, now we can use imsave or imread with out needing to preface it with
    # skimage.io like we do with np.array( [] )...
    # So let's use imread to read in week03_rgb.png and week03_msk.png: imread( <string of file.png> )
    rgb = imread("data/week04/week03_rgb.png")
    msk = imread("data/week04/week03_msk.png")
    print("rgb ", rgb.shape)
    print("msk ", msk.shape)
    # okay so let's look at the image shapes, this gets loaded in as a numpy array.
    # now let's look at the statistics of the RGB image.
    # We will use min() max() mean() and std()
    print(rgb.min(), rgb.max(), rgb.mean(), rgb.std())
    # let's discuss these values in the class.
    # what about if we define an axis to calculate the mean? So we had 1280*720*3, h*w*c
    # what do you think will happen if we calculate the min( axis=2 )? What is the second axis here?
    print(rgb.min(axis = 2).shape)
    # now we will use two colour space conversion. rgb to grey scale and rgb 2 Lab.
    # Does anyone know the problem with the RGB colour space?
    # colour spaces: https://scikit-image.org/docs/dev/api/skimage.color.html
    # First let's do a conversion to grey scale using rgb2gray, but first we need to import it!
    # We can import skimage.color but every time we wanted to use it we need to go import skimage.color.something.
    # This can become cumbersome, let's do something different to simplify it:
    # import skimage.color as skcol
    # So now we can do skcol.rgb2gray( rgb image )
    gry = skcol.rgb2gray(rgb)
    # and now save it using imsave( <savename.png>, image )
    imsave("data/week04/week03_rgb_to_gray.png", gry)
    # now let's do it with a slightly more interesting colour space.
    lab = skcol.rgb2lab(rgb)
    # The cool thing about the lab space is that it isn't additive! It uses the channels in a different way to RGB.
    # Channel 0 is the luminance, and 1,2 are the colours (chromanance).
    # So we can illuminate the image using channel 0 by adding or subtracting values then converting them back to
    # RGB. This is significantly more difficult in the RGB space.
    # Let's add 20 to the 0'th channel of the lab image then convert it back to RGB and save it.
    print(lab.shape)
    lab[:,:,0] += 20
    rgb0 = skcol.lab2rgb(lab)
    imsave("data/week04/rgb_luminance_20.png", rgb0)
    # now visually compare the two images.
    # This is the end of this weeks practical.
