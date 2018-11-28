import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate
from scipy.spatial import distance
import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap
from ipywidgets import FloatProgress
from IPython.display import display
import numba
import matplotlib.colors as mcolors

class Channel:
    """class for Channel objects"""
    def __init__(self,x,y,z,W,D):
        """initialize Channel object
        x, y, z  - coordinates of centerline
        W - channel width
        D - channel depth"""
        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D

class Cutoff:
    """class for Cutoff objects"""
    def __init__(self,x,y,z,W,D):
        """initialize Cutoff object
        x, y, z  - coordinates of centerline
        W - channel width
        D - channel depth"""
        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D

class ChannelBelt:
    """class for ChannelBelt objects"""
    def __init__(self, channels, cutoffs, cl_times, cutoff_times):
        """initialize ChannelBelt object
        channels - list of Channel objects
        cutoffs - list of Cutoff objects
        cl_times - list of ages of Channel objects
        cutoff_times - list of ages of Cutoff objects"""
        self.channels = channels
        self.cutoffs = cutoffs
        self.cl_times = cl_times
        self.cutoff_times = cutoff_times

    def migrate(self,nit,saved_ts,deltas,pad,crdist,Cf,kl,kv,dt,dens,*D):
        """function for computing migration rates along channel centerlines and moving the centerlines accordingly
        inputs:
        nit - number of iterations
        saved_ts - which time steps will be saved
        deltas - distance between nodes on centerline
        pad - padding (number of nodepoints along centerline)
        crdist - threshold distance at which cutoffs occur
        Cf - dimensionless Chezy friction factor
        kl - migration rate constant (m/s)
        kv - vertical slope-dependent erosion rate constant (m/s)
        dt - time step (s)
        dens - density of fluid (kg/m3)"""
        channel = self.channels[-1] # first channel is the same as last channel of input
        x = channel.x; y = channel.y; z = channel.z
        W = channel.W;
        if len(D)==0: 
            D = channel.D
        else:
            D = D[0]
        k = 1.0 # constant in HK equation
        xc = [] # initialize cutoff coordinates
        # determine age of last channel:
        if len(self.cl_times)>0:
            last_cl_time = self.cl_times[-1]
        else:
            last_cl_time = 0
        dx, dy, ds, s = compute_derivatives(x,y)
        slope = np.gradient(z)/ds
        # padding at the beginning can be shorter than padding at the downstream end:
        pad1 = int(pad/10.0)
        if pad1<5:
            pad1 = 5
        omega = -1.0 # constant in curvature calculation (Howard and Knutson, 1984)
        gamma = 2.5 # from Ikeda et al., 1981 and Howard and Knutson, 1984
        f = FloatProgress(min=1,max=nit) # progress bar
        display(f)
        for itn in range(nit): # main loop
            f.value += 1
            ns=len(x)
            dx, dy, ds, s, curv = compute_curvature(x,y)
            curv = W*curv # dimensionless curvature
            R0 = kl*curv # simple linear relationship between curvature and nominal migration rate
            alpha = k*2*Cf/D # exponent for convolution function G
            R1 = compute_migration_rate(pad,ns,ds,alpha,omega,gamma,R0)
            # calculate new centerline coordinates:
            dy_ds = dy[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
            dx_ds = dx[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
            # adjust x and y coordinates (this *is* the migration):
            x[pad1:ns-pad+1] = x[pad1:ns-pad+1] + R1[pad1:ns-pad+1]*dy_ds*dt  
            y[pad1:ns-pad+1] = y[pad1:ns-pad+1] - R1[pad1:ns-pad+1]*dx_ds*dt 
            # find and execute cutoffs:
            x,y,z,xc,yc,zc = cut_off_cutoffs(x,y,z,s,crdist,deltas) 
            dx, dy, ds, s = compute_derivatives(x,y) # recompute derivatives
            # resample centerline so that 'deltas' is roughly constant
            # [parametric spline representation of curve; note that there is *no* smoothing]
            tck, u = scipy.interpolate.splprep([x,y,z],s=0) 
            unew = np.linspace(0,1,1+s[-1]/deltas) # vector for resampling
            out = scipy.interpolate.splev(unew,tck) # resampling
            x, y, z = out[0], out[1], out[2] # assign new coordinate values
            dx, dy, ds, s = compute_derivatives(x,y) # recompute derivatives
            # incision:
            slope = np.gradient(z)/ds
            # slope-dependent erosion:
            z = z + kv*dens*9.81*D*slope*dt         
            if len(xc)>0: # save cutoff data
                self.cutoff_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                cutoff = Cutoff(xc,yc,zc,W,D) # create cutoff object
                self.cutoffs.append(cutoff)
            # saving centerlines:
            if np.mod(itn,saved_ts)==0:
                self.cl_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                channel = Channel(x,y,z,W,D) # create channel object
                self.channels.append(channel)

    def plot(self,plot_type,pb_age,ob_age,*end_time):
        """plot ChannelBelt object
        plot_type - can be either 'strat' (for stratigraphic plot) or 'morph' (for morphologic plot)
        pb_age - age of point bars (in years) at which they get covered by vegetation
        ob_age - age of oxbow lakes (in years) at which they get covered by vegetation
        end_time (optional) - age of last channel to be plotted (in years)"""
        cot = np.array(self.cutoff_times)
        sclt = np.array(self.cl_times)
        if len(end_time)>0:
            cot = cot[cot<=end_time]
            sclt = sclt[sclt<=end_time]
        times = np.sort(np.hstack((cot,sclt)))
        times = np.unique(times)
        order = 0 # variable for ordering objects in plot
        # set up min and max x and y coordinates of the plot:
        xmin = np.min(self.channels[0].x)
        xmax = np.max(self.channels[0].x)
        ymax = 0
        for i in range(len(self.channels)):
            ymax = max(ymax, np.max(np.abs(self.channels[i].y)))
        ymax = ymax+2*self.channels[0].W # add a bit of space on top and bottom
        ymin = -1*ymax
        # size figure so that its size matches the size of the model:
        fig = plt.figure(figsize=(20,(ymax-ymin)*20/(xmax-xmin))) 
        if plot_type == 'morph':
            pb_crit = len(times[times<times[-1]-pb_age])/float(len(times))
            ob_crit = len(times[times<times[-1]-ob_age])/float(len(times))
            green = (106/255.0,159/255.0,67/255.0) # vegetation color
            pb_color = (189/255.0,153/255.0,148/255.0) # point bar color
            ob_color = (15/255.0,58/255.0,65/255.0) # oxbow color
            pb_cmap = make_colormap([green,green,pb_crit,green,pb_color,1.0,pb_color]) # colormap for point bars
            ob_cmap = make_colormap([green,green,ob_crit,green,ob_color,1.0,ob_color]) # colormap for oxbows
            plt.fill([xmin,xmax,xmax,xmin],[ymin,ymin,ymax,ymax],color=(106/255.0,159/255.0,67/255.0))
        for i in range(0,len(times)):
            if times[i] in sclt:
                ind = np.where(sclt==times[i])[0][0]
                x1 = self.channels[ind].x
                y1 = self.channels[ind].y
                W = self.channels[ind].W
                xm, ym = get_channel_banks(x1,y1,W)
                if plot_type == 'morph':
                    if times[i]>times[-1]-pb_age:
                        plt.fill(xm,ym,facecolor=pb_cmap(i/float(len(times)-1)),edgecolor='k',linewidth=0.2)
                    else:
                        plt.fill(xm,ym,facecolor=pb_cmap(i/float(len(times)-1)))
                else:
                    order = order+1
                    plt.fill(xm,ym,sns.xkcd_rgb["light tan"],edgecolor='k',linewidth=0.25,zorder=order)
            if times[i] in cot:
                ind = np.where(cot==times[i])[0][0]
                for j in range(0,len(self.cutoffs[ind].x)):
                    x1 = self.cutoffs[ind].x[j]
                    y1 = self.cutoffs[ind].y[j]
                    xm, ym = get_channel_banks(x1,y1,self.cutoffs[ind].W)
                    if plot_type == 'morph':
                        plt.fill(xm,ym,color=ob_cmap(i/float(len(times)-1)))
                    else:
                        order = order+1
                        plt.fill(xm,ym,sns.xkcd_rgb["ocean blue"],edgecolor='k',linewidth=0.25,zorder=order)
        x1 = self.channels[len(sclt)-1].x
        y1 = self.channels[len(sclt)-1].y
        xm, ym = get_channel_banks(x1,y1,self.channels[len(sclt)-1].W)
        order = order+1
        plt.fill(xm,ym,color=(16/255.0,73/255.0,90/255.0),zorder=order) #,edgecolor='k')
        plt.axis('equal')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        return fig

    def create_movie(self,xmin,xmax,plot_type,filename,dirname,pb_age,ob_age,scale,*end_time):
        """method for creating movie frames (PNG files) that capture the plan-view evolution of a channel belt through time
        movie has to be assembled from the PNG file after this method is applied
        xmin - value of x coodinate on the left side of frame
        xmax - value of x coordinate on right side of frame
        plot_type = - can be either 'strat' (for stratigraphic plot) or 'morph' (for morphologic plot)
        filename - first few characters of the output filenames
        dirname - name of directory where output files should be written
        pb_age - age of point bars (in years) at which they get covered by vegetation (if the 'morph' option is used for 'plot_type')
        ob_age - age of oxbow lakes (in years) at which they get covered by vegetation (if the 'morph' option is used for 'plot_type')
        scale - scaling factor (e.g., 2) that determines how many times larger you want the frame to be, compared to the default scaling of the figure
        """
        sclt = np.array(self.cl_times)
        if len(end_time)>0:
            sclt = sclt[sclt<=end_time]
        channels = self.channels[:len(sclt)]
        ymax = 0
        for i in range(len(channels)):
            ymax = max(ymax, np.max(np.abs(channels[i].y)))
        ymax = ymax+2*channels[0].W # add a bit of space on top and bottom
        ymin = -1*ymax
        for i in range(0,len(sclt)):
            fig = self.plot(plot_type,pb_age,ob_age,sclt[i])
            fig_height = scale*fig.get_figheight()
            fig_width = (xmax-xmin)*fig_height/(ymax-ymin)
            fig.set_figwidth(fig_width)
            fig.set_figheight(fig_height)
            fig.gca().set_xlim(xmin,xmax)
            fig.gca().set_xticks([])
            fig.gca().set_yticks([])
            plt.plot([xmin+200, xmin+200+5000],[ymin+200, ymin+200], 'k', linewidth=2)
            plt.text(xmin+200+2000, ymin+200+100, '5 km', fontsize=14)
            fname = dirname+filename+'%03d.png'%(i)
            fig.savefig(fname, bbox_inches='tight')
            plt.close()

def generate_initial_channel(W,D,Sl,deltas,pad,n_bends):
    """generate straight Channel object with some noise added that can serve
    as input for initializing a ChannelBelt object
    W - channel width
    D - channel depth
    Sl - channel gradient
    deltas - distance between nodes on centerline
    pad - padding (number of nodepoints along centerline)
    n_bends - approximate number of bends to be simulated"""
    noisy_len = n_bends*10*W/2.0 # length of noisy part of initial centerline
    pad1 = int(pad/10.0) # padding at upstream end can be shorter than padding on downstream end
    if pad1<5:
        pad1 = 5
    x = np.linspace(0, noisy_len+(pad+pad1)*deltas, int(noisy_len/deltas+pad+pad1)+1) # x coordinate
    y = 10.0 * (2*np.random.random_sample(int(noisy_len/deltas)+1,)-1)
    y = np.hstack((np.zeros((pad1),),y,np.zeros((pad),))) # y coordinate
    deltaz = Sl * deltas*(len(x)-1)
    z = np.linspace(0,deltaz,len(x))[::-1] # z coordinate
    return Channel(x,y,z,W,D)

@numba.jit(nopython=True) # use Numba to speed up the heaviest computation
def compute_migration_rate(pad,ns,ds,alpha,omega,gamma,R0):
    """compute migration rate as weighted sum of upstream curvatures
    pad - padding (number of nodepoints along centerline)
    ns - number of points in centerline
    ds - distances between points in centerline
    omega - constant in HK model
    gamma - constant in HK model
    R0 - nominal migration rate (dimensionless curvature * migration rate constant)"""
    R1 = np.zeros(ns) # preallocate adjusted channel migration rate
    pad1 = int(pad/10.0) # padding at upstream end can be shorter than padding on downstream end
    if pad1<5:
        pad1 = 5
    for i in range(pad1,ns-pad):
        si2 = np.cumsum(ds[i:0:-1]) # distance along centerline, backwards from current point
        G = np.exp(-alpha*si2) # convolution vector
        R1[i] = omega*R0[i] + gamma*np.sum(R0[i:0:-1]*G)/np.sum(G) # main equation
    return R1

def compute_derivatives(x,y):
    """function for computing first derivatives of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ds = np.sqrt(dx**2+dy**2)
    s = np.cumsum(ds)
    return dx, dy, ds, s

def compute_curvature(x,y):
    """function for computing first derivatives and curvature of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve
    curvature - curvature of the curve (in 1/units of x and y)"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ds = np.sqrt(dx**2+dy**2)
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy - dy*ddx) / ((dx**2 + dy**2)**1.5)
    s = np.cumsum(ds)
    return dx, dy, ds, s, curvature

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    [from: https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale]
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def kth_diag_indices(a,k):
    """function for finding diagonal indices with k offset (from Stackexchange)"""
    rows, cols = np.diag_indices_from(a)
    if k<0:
        return rows[:k], cols[-k:]
    elif k>0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols
    
def find_cutoffs(x,y,crdist,deltas):
    """function for identifying locations of cutoffs along a centerline
    and the indices of the segments that will become part of the oxbows
    x,y - coordinates of centerline
    crdist - critical cutoff distance
    deltas - distance between neighboring points along the centerline"""
    diag_blank_width = int((crdist+20*deltas)/deltas)
    # distance matrix for centerline points:
    dist = distance.cdist(np.array([x,y]).T,np.array([x,y]).T)
    dist[dist>crdist] = np.NaN # set all values that are larger than the cutoff threshold to NaN
    # set matrix to NaN along the diagonal zone:
    for k in range(-diag_blank_width,diag_blank_width+1):
        rows, cols = kth_diag_indices(dist,k)
        dist[rows,cols] = np.NaN
    i1, i2 = np.where(~np.isnan(dist))
    ind1 = i1[np.where(i1<i2)[0]] # get rid of unnecessary indices
    ind2 = i2[np.where(i1<i2)[0]] # get rid of unnecessary indices
    return ind1, ind2 # return indices of cutoff points and cutoff coordinates

def cut_off_cutoffs(x,y,z,s,crdist,deltas):
    """function for executing cutoffs - removing oxbows from centerline and storing cutoff coordinates
    x,y - coordinates of centerline
    crdist - critical cutoff distance
    deltas - distance between neighboring points along the centerline
    outputs:
    x,y,z - updated coordinates of centerline
    xc, yc, zc - lists with coordinates of cutoff segments"""
    xc = []
    yc = []
    zc = []
    ind1, ind2 = find_cutoffs(x,y,crdist,deltas) # initial check for cutoffs
    while len(ind1)>0:
        xc.append(x[ind1[0]:ind2[0]+1]) # x coordinates of cutoff
        yc.append(y[ind1[0]:ind2[0]+1]) # y coordinates of cutoff
        zc.append(z[ind1[0]:ind2[0]+1]) # z coordinates of cutoff
        x = np.hstack((x[:ind1[0]+1],x[ind2[0]:])) # x coordinates after cutoff
        y = np.hstack((y[:ind1[0]+1],y[ind2[0]:])) # y coordinates after cutoff
        z = np.hstack((z[:ind1[0]+1],z[ind2[0]:])) # z coordinates after cutoff
        ind1, ind2 = find_cutoffs(x,y,crdist,deltas)       
    return x,y,z,xc,yc,zc

def get_channel_banks(x,y,W):
    """function for finding coordinates of channel banks, given a centerline and a channel width
    x,y - coordinates of centerline
    W - channel width
    outputs:
    xm, ym - coordinates of channel banks (both left and right banks)"""
    x1 = x.copy()
    y1 = y.copy()
    x2 = x.copy()
    y2 = y.copy()
    ns = len(x)
    dx = np.diff(x); dy = np.diff(y) 
    ds = np.sqrt(dx**2+dy**2)
    x1[:-1] = x[:-1] + 0.5*W*np.diff(y)/ds
    y1[:-1] = y[:-1] - 0.5*W*np.diff(x)/ds
    x2[:-1] = x[:-1] - 0.5*W*np.diff(y)/ds
    y2[:-1] = y[:-1] + 0.5*W*np.diff(x)/ds
    x1[ns-1] = x[ns-1] + 0.5*W*(y[ns-1]-y[ns-2])/ds[ns-2]
    y1[ns-1] = y[ns-1] - 0.5*W*(x[ns-1]-x[ns-2])/ds[ns-2]
    x2[ns-1] = x[ns-1] - 0.5*W*(y[ns-1]-y[ns-2])/ds[ns-2]
    y2[ns-1] = y[ns-1] + 0.5*W*(x[ns-1]-x[ns-2])/ds[ns-2]
    xm = np.hstack((x1,x2[::-1]))
    ym = np.hstack((y1,y2[::-1]))
    return xm, ym
