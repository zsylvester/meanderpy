# meanderpy
## Simple model of meander migration

'meanderpy' is a Python module that implements a simple numerical model of meandering, the one described by Howard & Knutson in their 1984 paper ["Sufficient Conditions for River Meandering: A Simulation Approach"](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/WR020i011p01659). This is a kinematic model that is based on computing migration rate as the weighted sum of upstream curvatures; flow velocity does not enter the equation. Curvature is transformed into a 'nominal migration rate' through multiplication with a migration rate (or erodibility) constant; in the  Howard & Knutson (1984) paper this is a nonlinear relationship based on field observations that suggested a complex link between curvature and migration rate. In the 'meanderpy' module we use a simple linear relationship between the nominal migration rate and curvature, as recent work using time-lapse satellite imagery suggests that high curvatures result in high migration rates (Sylvester et al., in review).

<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_sketch.png" width="600">

The sketch above shows the three 'meanderpy' components: channel, cutoff, channel belt. These are implemented as classes; a 'Channel' and a 'Cutoff' are defined by their width, depth, and x,y,z centerline coordinates, and a 'ChannelBelt' is a collection of channels and cutoffs. In addition, the 'ChannelBelt' object also has a 'cl_times' and a 'cutoff_times' attribute that specify the age of the channels and the cutoffs. This age is relative to the start time of the simulation (= the first channel, age = 0.0).

The core functionality of 'meanderpy' is built into the 'migrate' method of the 'ChannelBelt' class. This is the function that computes migration rates and moves the channel centerline to its new position. The last Channel of a ChannelBelt can be further migrated through applying the 'migrate' method to the ChannelBelt instance.

The initial Channel object can be created using the 'generate_initial_channel' function. This creates a straight line, with some noise added. However, a Channel can be created (and then used as the first channel in a ChannelBelt) using any set of x,y,z,W,D variables.

ChannelBelt objects can be visualized using the 'plot' method. This creates a map of all the channels and cutoffs in the channel belt; there are two styles of plotting: a 'stratigraphic' view and a 'morphologic' view (see below). The morphologic view tries to account for the fact that older point bars and oxbow lakes tend to be gradually covered with vegetation. 

<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_strat_vs_morph.png" width="1000">

