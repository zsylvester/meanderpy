# meanderpy
## Simple model of meander migration

'meanderpy' is a Python module that implements the simplest numerical model of meandering, the one described by Howard & Knutson in their 1984 paper "Sufficient Conditions for River Meandering: A Simulation Approach". This is a kinematic model that is based on computing migration rate as the weighted sum of upstream curvatures; flow velocity does not enter the equation. Curvature is transformed into a 'nominal migration rate' through multiplication with a migration rate (or erodibility) constant; in the original Howard & Knutson paper this is a nonlinear relationship based on field observations that suggested a complex link between curvature and migration rate. In the 'meanderpy' module we use a simple linear relationship between the nominal migration rate and curvature, for a variety of reasons.

<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_sketch.png" width="600">

The sketch above shows the three 'meanderpy' components: channel, cutoff, channel belt. These are implemented as classes; a 'Channel' and a 'Cutoff' are defined by their width, depth, and x,y,z centerline coordinates, and a 'ChannelBelt' is a collection of channels and cutoffs. In addition, the 'ChannelBelt' object also has a 'cl_times' and a 'cutoff_times' attribute that specify the age of the channels and the cutoffs.
