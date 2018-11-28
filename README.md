<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_logo.svg" width="300">

## Simple model of meander migration

'meanderpy' is a Python module that implements a simple numerical model of meandering, the one described by Howard & Knutson in their 1984 paper ["Sufficient Conditions for River Meandering: A Simulation Approach"](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/WR020i011p01659). This is a kinematic model that is based on computing migration rate as the weighted sum of upstream curvatures; flow velocity does not enter the equation. Curvature is transformed into a 'nominal migration rate' through multiplication with a migration rate (or erodibility) constant; in the  Howard & Knutson (1984) paper this is a nonlinear relationship based on field observations that suggested a complex link between curvature and migration rate. In the 'meanderpy' module we use a simple linear relationship between the nominal migration rate and curvature, as recent work using time-lapse satellite imagery suggests that high curvatures result in high migration rates (Sylvester et al., in review).

<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_sketch.png" width="600">

The sketch above shows the three 'meanderpy' components: channel, cutoff, channel belt. These are implemented as classes; a 'Channel' and a 'Cutoff' are defined by their width, depth, and x,y,z centerline coordinates, and a 'ChannelBelt' is a collection of channels and cutoffs. In addition, the 'ChannelBelt' object also has a 'cl_times' and a 'cutoff_times' attribute that specify the age of the channels and the cutoffs. This age is relative to the start time of the simulation (= the first channel, age = 0.0).

The initial Channel object can be created using the 'generate_initial_channel' function. This creates a straight line, with some noise added. However, a Channel can be created (and then used as the first channel in a ChannelBelt) using any set of x,y,z,W,D variables.

```python
ch = mp.generate_initial_channel(W,D,Sl,deltas,pad,n_bends) # initialize channel
chb = mp.ChannelBelt(channels=[ch],cutoffs=[],cl_times=[0.0],cutoff_times=[]) # create channel belt object
```

A reasonable set of input parameters are as follows:

```python
W = 200.0                    # channel width (m)
D = 16.0                     # channel depth (m)
pad = 100                    # padding (number of nodepoints along centerline)
deltas = 50.0                # sampling distance along centerline
nit = 1500                   # number of iterations
Cf = 0.03                    # dimensionless Chezy friction factor
crdist = W                   # threshold distance at which cutoffs occur
kl = 60.0/(365*24*60*60.0)   # migration rate constant (m/s)
kv =  3*10*5.0E-13           # vertical slope-dependent erosion rate constant (m/s)
dt = 0.1*(365*24*60*60.0)    # time step (s)
dens = 1000                  # density of water (kg/m3)
saved_ts = 20                # which time steps will be saved
n_bends = 30                 # approximate number of bends you want to model
Sl = 0.0                     # initial slope (setting this to non-zero results in instabilities in long runs)
```

The core functionality of 'meanderpy' is built into the 'migrate' method of the 'ChannelBelt' class. This is the function that computes migration rates and moves the channel centerline to its new position. The last Channel of a ChannelBelt can be further migrated through applying the 'migrate' method to the ChannelBelt instance.

```python
chb.migrate(nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens) # channel migration
```

ChannelBelt objects can be visualized using the 'plot' method. This creates a map of all the channels and cutoffs in the channel belt; there are two styles of plotting: a 'stratigraphic' view and a 'morphologic' view (see below). The morphologic view tries to account for the fact that older point bars and oxbow lakes tend to be gradually covered with vegetation. 

```python
# migrate an additional 1000 iterations and plot results
chb.migrate(1000,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens)
fig = chb.plot('strat',20,60)
```

<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_strat_vs_morph.png" width="1000">

A series of movie frames (in PNG format) can be created using the 'create_movie' method:

```python
create_movie(xmin,xmax,plot_type,filename,dirname,pb_age,ob_age,scale,end_time)
```
