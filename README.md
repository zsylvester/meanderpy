# meanderpy
Simple model of meander migration

'meanderpy' is a Python module that implements the simplest numerical model of meandering, the one described by Howard & Knutson in their 1984 paper "Sufficient Conditions for River Meandering: A Simulation Approach". This is a kinematic model that is based on computing migration rate as the weighted sum of upstream curvatures; flow velocity does not enter the equation. Curvature is transformed into a 'nominal migration rate' through multiplication with a migration rate (or erodibility) constant; in the original Howard & Knutson paper this is a nonlinear relationship based on field observations that suggested a complex link between curvature and migration rate. In the 'meanderpy' module we use a simple linear relationship between the nominal migration rate and curvature, for a variety of reasons.

<img src="https://github.com/zsylvester/meanderpy/blob/master/meanderpy_sketch.png" width="800">

Sketch showing the three 'meanderpy' components (classes): channel, cutoff, channel belt
