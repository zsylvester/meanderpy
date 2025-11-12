import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.spatial import distance
from scipy import ndimage
from PIL import Image, ImageDraw
from skimage import measure
from skimage import morphology
# from skimage import filters
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import time, sys
import numba
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import trange
import h5py
from scipy.signal import savgol_filter


class Channel:
    """class for Channel objects"""

    def __init__(self,x,y,z,W,D):
        """
        Initialize Channel object.

        Parameters
        ----------
        x : array_like
            x-coordinate of centerline.
        y : array_like
            y-coordinate of centerline.
        z : array_like
            z-coordinate of centerline.
        W : float
            Channel width.
        D : float
            Channel depth.
        """

        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D

class Cutoff:
    """class for Cutoff objects"""

    def __init__(self,x,y,z,W,D):
        """
        Initialize Cutoff object.

        Parameters
        ----------
        x : array_like
            x-coordinate of centerline.
        y : array_like
            y-coordinate of centerline.
        z : array_like
            z-coordinate of centerline.
        W : float
            Channel width.
        D : float
            Channel depth.
        """

        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D

class ChannelBelt3D:
    """class for 3D models of channel belts"""

    def __init__(self, model_type, topo, strat, facies, facies_code, dx, channels):
        """
        Initialize ChannelBelt3D object.

        Parameters
        ----------
        model_type : str
            Type of model to be built; can be either 'fluvial' or 'submarine'.
        topo : numpy.ndarray
            Set of topographic surfaces (3D numpy array).
        strat : numpy.ndarray
            Set of stratigraphic surfaces (3D numpy array).
        facies : numpy.ndarray
            Facies volume (3D numpy array).
        facies_code : dict
            Dictionary of facies codes, e.g. {0:'oxbow', 1:'point bar', 2:'levee'}.
        dx : float
            Gridcell size (m).
        channels : list
            List of channel objects that form 3D model.
        """

        self.model_type = model_type
        self.topo = topo
        self.strat = strat
        self.facies = facies
        self.facies_code = facies_code
        self.dx = dx
        self.channels = channels

    def plot_xsection(self, xsec, colors, ve):
        """
        Method for plotting a cross section through a 3D model; also plots map of 
        basal erosional surface and map of final geomorphic surface.

        Parameters
        ----------
        xsec : int
            Location of cross section along the x-axis (in pixel/voxel coordinates).
        colors : list of tuple
            List of RGB values that define the colors for different facies.
        ve : float
            Vertical exaggeration.

        Returns
        -------
        fig1 : matplotlib.figure.Figure
            Figure handle for the cross section plot.
        fig2 : matplotlib.figure.Figure
            Figure handle for the final geomorphic surface plot.
        fig3 : matplotlib.figure.Figure
            Figure handle for the basal erosional surface plot.
        """

        strat = self.strat
        dx = self.dx
        fig1 = plt.figure(figsize=(20,5))
        ax1 = fig1.add_subplot(111)
        r,c,ts = np.shape(strat)
        Xv = dx * np.arange(0,r)
        for i in range(0,ts-1,2):
            X1 = np.concatenate((Xv, Xv[::-1]))  
            Y1 = np.concatenate((strat[:,xsec,i], strat[::-1,xsec,i+1])) 
            Y2 = np.concatenate((strat[:,xsec,i+1], strat[::-1,xsec,i+2]))
            # Y3 = np.concatenate((strat[:,xsec,i+2], strat[::-1,xsec,i+3]))
            if self.model_type == 'submarine':
                ax1.fill(X1, Y1, facecolor=colors[0], linewidth=0.5, edgecolor=[0,0,0]) # channel sand
                ax1.fill(X1, Y2, facecolor=colors[1], linewidth=0.5, edgecolor=[0,0,0]) # levee mud
            if self.model_type == 'fluvial':
                ax1.fill(X1, Y1, facecolor=colors[0], linewidth=0.5, edgecolor=[0,0,0]) # channel sand
                ax1.fill(X1, Y2, facecolor=colors[1], linewidth=0.5, edgecolor=[0,0,0]) # levee mud
                # ax1.fill(X1, Y3, facecolor=colors[2], linewidth=0.5) # channel sand
        ax1.set_xlim(0,dx*(r-1))
        ax1.set_aspect(ve, adjustable='datalim')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.contourf(strat[:,:,ts-1],100,cmap='viridis')
        ax2.contour(strat[:,:,ts-1],100,colors='k',linestyles='solid',linewidths=0.1,alpha=0.4)
        ax2.plot([xsec, xsec],[0,r],'k',linewidth=2)
        ax2.axis([0,c,0,r])
        ax2.set_aspect('equal', adjustable='box')        
        ax2.set_title('final geomorphic surface')
        ax2.tick_params(bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.contourf(strat[:,:,0],100,cmap='viridis')
        ax3.contour(strat[:,:,0],100,colors='k',linestyles='solid',linewidths=0.1,alpha=0.4)
        ax3.plot([xsec, xsec],[0,r],'k',linewidth=2)
        ax3.axis([0,c,0,r])
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title('basal erosional surface')
        ax3.tick_params(bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        return fig1, fig2, fig3

class ChannelBelt:
    """class for ChannelBelt objects"""

    def __init__(self, channels, cutoffs, cl_times, cutoff_times):
        """
        Initialize ChannelBelt object.

        Parameters
        ----------
        channels : list of Channel
            List of Channel objects.
        cutoffs : list of Cutoff
            List of Cutoff objects.
        cl_times : list of float
            List of ages of Channel objects (in years).
        cutoff_times : list of float
            List of ages of Cutoff objects.
        """

        self.channels = channels
        self.cutoffs = cutoffs
        self.cl_times = cl_times
        self.cutoff_times = cutoff_times

    def migrate(self, nit, saved_ts, deltas, pad, crdist, depths, Cfs, kl, kv, dt, dens, autoaggradation=True, Scr=0.001, t1=None, t2=None, t3=None, aggr_factor=None):
        """
        Compute migration rates along channel centerlines and move the centerlines accordingly.

        Parameters
        ----------
        nit : int
            Number of iterations.
        saved_ts : int
            Which time steps will be saved; e.g., if saved_ts = 10, every tenth time step will be saved.
        deltas : float
            Distance between nodes on centerline.
        pad : int
            Padding (number of nodepoints along centerline).
        crdist : float
            Threshold distance at which cutoffs occur.
        depths : array_like
            Array of channel depths (can vary across iterations).
        Cfs : array_like
            Array of dimensionless Chezy friction factors (can vary across iterations).
        kl : float
            Migration rate constant (m/s).
        kv : float
            Vertical slope-dependent erosion rate constant (m/s).
        dt : float
            Time step (s).
        dens : float
            Density of fluid (kg/m^3).
        autoaggradation : bool, optional
            If True, autoaggradation is applied. Default is True.
        Scr : float, optional
            Critical slope for autoaggradation. Default is 0.001.
        t1 : int, optional
            Time step when incision starts. Default is None.
        t2 : int, optional
            Time step when lateral migration starts. Default is None.
        t3 : int, optional
            Time step when aggradation starts. Default is None.
        aggr_factor : float, optional
            Aggradation factor. Default is None.
        """

        channel = self.channels[-1] # first channel is the same as last channel of input
        x = channel.x; y = channel.y; z = channel.z
        W = channel.W
        D = channel.D
        k = 1.0 # constant in HK equation
        xc = [] # initialize cutoff coordinates
        # determine age of last channel:
        if len(self.cl_times)>0:
            last_cl_time = self.cl_times[-1]
        else:
            last_cl_time = 0
        dx, dy, dz, ds, s = compute_derivatives(x,y,z)
        slope = np.gradient(z)/ds
        # padding at the beginning can be shorter than padding at the downstream end:
        pad1 = int(pad/10.0)
        if pad1<5:
            pad1 = 5
        for itn in trange(nit): # main loop
            D = depths[itn]
            Cf = Cfs[itn]
            x, y = migrate_one_step(x,y,z,W,kl,dt,k,Cf,D,pad,pad1)
            # x, y = migrate_one_step_w_bias(x,y,z,W,kl,dt,k,Cf,D,pad,pad1)
            x,y,z,xc,yc,zc = cut_off_cutoffs(x,y,z,s,crdist,deltas) # find and execute cutoffs
            x,y,z,dx,dy,dz,ds,s = resample_centerline(x,y,z,deltas) # resample centerline
            z = savgol_filter(z, 21, 2) # filter z-values - needed for autoaggradation
            slope = np.gradient(z)/ds # positive number
            if autoaggradation:
                if np.max(slope) > 0.001:
                    R = 1.65; C = 0.1 # parameters for autoaggradation
                    z = z + kv*dens*9.81*R*C*D*(Scr-slope)*dt # autoaggradation
            else:
                # for itn<=t1, z is unchanged; for itn>t1:
                if (itn>t1) & (itn<=t2): # incision
                    if np.min(np.abs(slope))!=0: # if slope is not zero
                        z = z + kv*dens*9.81*D*slope*dt
                    else:
                        z = z - kv*dens*9.81*D*dt*0.05 # if slope is zero
                if (itn>t2) & (itn<=t3): # lateral migration
                    if np.min(np.abs(slope))!=0: # if slope is not zero
                        # use the median slope to counterbalance incision:
                        z = z + kv*dens*9.81*D*slope*dt - kv*dens*9.81*D*np.median(slope)*dt
                    else:
                        z = z # no change in z
                if (itn>t3): # aggradation
                    if np.min(np.abs(slope))!=0: # if slope is not zero
                        # 'aggr_factor' should be larger than 1 so that this leads to overall aggradation:
                        z = z + kv*dens*9.81*D*slope*dt - aggr_factor*kv*dens*9.81*D*np.mean(slope)*dt
                    else:
                        z = z + aggr_factor*dt
            if len(xc)>0: # save cutoff data
                self.cutoff_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                cutoff = Cutoff(xc,yc,zc,W,D) # create cutoff object
                self.cutoffs.append(cutoff)
            # saving centerlines (with the exception of first channel):
            if (np.mod(itn, saved_ts) == 0) and (itn > 0):
                self.cl_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                channel = Channel(x,y,z,W,D) # create channel object
                self.channels.append(channel)

    def plot(self, plot_type, pb_age, ob_age, end_time, n_channels):
        """
        Method for plotting ChannelBelt object.

        Parameters
        ----------
        plot_type : str
            Can be either 'strat' (for stratigraphic plot), 'morph' (for morphologic plot), or 'age' (for age plot).
        pb_age : int
            Age of point bars (in years) at which they get covered by vegetation.
        ob_age : int
            Age of oxbow lakes (in years) at which they get covered by vegetation.
        end_time : int
            Age of last channel to be plotted (in years).
        n_channels : int
            Total number of channels (used in 'age' plots; can be larger than number of channels being plotted).

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle to the figure.
        """

        cot = np.array(self.cutoff_times)
        sclt = np.array(self.cl_times)
        if end_time>0:
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
        if plot_type == 'age':
            age_cmap = cm.get_cmap('magma', n_channels)
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
                if plot_type == 'strat':
                    order += 1
                    plt.fill(xm, ym, 'xkcd:light tan', edgecolor='k', linewidth=0.25, zorder=order)
                    # plt.fill(xm, ym, 'xkcd:light tan', edgecolor='none', linewidth=0.25, zorder=order)
                if plot_type == 'age':
                    order += 1
                    plt.fill(xm,ym,facecolor=age_cmap(i/float(n_channels-1)),edgecolor='k',linewidth=0.1,zorder=order)
            if times[i] in cot:
                ind = np.where(cot==times[i])[0][0]
                for j in range(0,len(self.cutoffs[ind].x)):
                    x1 = self.cutoffs[ind].x[j]
                    y1 = self.cutoffs[ind].y[j]
                    xm, ym = get_channel_banks(x1,y1,self.cutoffs[ind].W)
                    if plot_type == 'morph':
                        plt.fill(xm,ym,color=ob_cmap(i/float(len(times)-1)))
                    if plot_type == 'strat':
                        order = order+1
                        plt.fill(xm, ym, 'xkcd:ocean blue', edgecolor='k', linewidth=0.25, zorder=order)
                        # plt.fill(xm, ym, 'xkcd:ocean blue', edgecolor='none', linewidth=0.25, zorder=order)
                    if plot_type == 'age':
                        order += 1
                        plt.fill(xm, ym, 'xkcd:sea blue', edgecolor='k', linewidth=0.1, zorder=order)

        x1 = self.channels[len(sclt)-1].x
        y1 = self.channels[len(sclt)-1].y
        xm, ym = get_channel_banks(x1,y1,self.channels[len(sclt)-1].W)
        order = order+1
        if plot_type == 'age':
            plt.fill(xm, ym, color='xkcd:sea blue', zorder=order, edgecolor='k', linewidth=0.1)
        else:
            plt.fill(xm, ym, color=(16/255.0,73/255.0,90/255.0), edgecolor='none', zorder=order) #,edgecolor='k')
        plt.axis('equal')
        # plt.xlim(xmin,xmax)
        # plt.ylim(ymin,ymax)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        return fig

    def create_movie(self, xmin, xmax, plot_type, filename, dirname, pb_age, ob_age, end_time, n_channels):
        """
        Method for creating movie frames (PNG files) that capture the plan-view evolution of a channel belt through time.
        The movie has to be assembled from the PNG files after this method is applied.

        Parameters
        ----------
        xmin : float
            Value of x coordinate on the left side of the frame.
        xmax : float
            Value of x coordinate on the right side of the frame.
        plot_type : str
            Plot type; can be either 'strat' (for stratigraphic plot) or 'morph' (for morphologic plot).
        filename : str
            First few characters of the output filenames.
        dirname : str
            Name of the directory where output files should be written.
        pb_age : int
            Age of point bars (in years) at which they get covered by vegetation (if the 'morph' option is used for 'plot_type').
        ob_age : int
            Age of oxbow lakes (in years) at which they get covered by vegetation (if the 'morph' option is used for 'plot_type').
        end_time : float or list of float
            Time at which the simulation should be stopped.
        n_channels : int
            Total number of channels + cutoffs for which the simulation is run (usually it is len(chb.cutoffs) + len(chb.channels)). Used when plot_type = 'age'.

        """

        sclt = np.array(self.cl_times)
        if type(end_time) != list:
            sclt = sclt[sclt<=end_time]
        channels = self.channels[:len(sclt)]
        ymax = 0
        for i in range(len(channels)):
            ymax = max(ymax, np.max(np.abs(channels[i].y)))
        ymax = ymax+2*channels[0].W # add a bit of space on top and bottom
        ymin = -1*ymax
        for i in range(0,len(sclt)):
            fig = self.plot(plot_type, pb_age, ob_age, sclt[i], n_channels)
            scale = 1
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

def build_3d_model(chb, model_type, h_mud, h, w, dx, delta_s, dt, starttime, endtime, diff_scale, v_fine, v_coarse, xmin=None, xmax=None, ymin=None, ymax=None, bth=None, dcr=None):
    """
    Build 3D model from set of centerlines (that are part of a ChannelBelt object).

    Parameters
    ----------
    model_type : str
        Model type ('fluvial' or 'submarine').
    h_mud : float
        Maximum thickness of overbank deposit.
    h : float
        Channel depth.
    w : float
        Channel width.
    dx : float
        Cell size in x and y directions.
    delta_s : float
        Sampling distance along centerlines.
    starttime : float
        Age of centerline that will be used as the first centerline in the model.
    endtime : float
        Age of centerline that will be used as the last centerline in the model.
    xmin : float, optional
        Minimum x coordinate that defines the model domain; if xmin is set to zero, 
        a plot of the centerlines is generated and the model domain has to be defined by clicking its upper left and lower right corners.
    xmax : float, optional
        Maximum x coordinate that defines the model domain.
    ymin : float, optional
        Minimum y coordinate that defines the model domain.
    ymax : float, optional
        Maximum y coordinate that defines the model domain.
    diff_scale : float
        Diffusion length scale (for overbank deposition).
    v_fine : float
        Deposition rate of fine sediment, in m/year (for overbank deposition).
    v_coarse : float
        Deposition rate of coarse sediment, in m/year (for overbank deposition).
    bth : float, optional
        Thickness of channel sand (only used in submarine models).
    dcr : float, optional
        Critical channel depth where sand thickness goes to zero (only used in submarine models).

    Returns
    -------
    chb_3d : ChannelBelt3D
        A ChannelBelt3D object.
    xmin : float
        Minimum x coordinate that defines the model domain.
    xmax : float
        Maximum x coordinate that defines the model domain.
    ymin : float
        Minimum y coordinate that defines the model domain.
    ymax : float
        Maximum y coordinate that defines the model domain.
    """

    sclt = np.array(chb.cl_times)
    ind1 = np.where(sclt >= starttime)[0][0] 
    ind2 = np.where(sclt <= endtime)[0][-1]
    sclt = sclt[ind1:ind2+1]
    channels = chb.channels[ind1:ind2+1]
    cot = np.array(chb.cutoff_times)
    if (len(cot)>0) and (len(np.where(cot >= starttime)[0])>0) and (len(np.where(cot <= endtime)[0])>0):
        cfind1 = np.where(cot >= starttime)[0][0] 
        cfind2 = np.where(cot <= endtime)[0][-1]
        cot = cot[cfind1:cfind2+1]
    else:
        cot = []
    n_steps = len(sclt) # number of events
    if xmin is None: # plot centerlines and define model domain
        plt.figure(figsize=(15,10))
        for i in range(len(channels)): # plot centerlines
            plt.plot(chb.channels[i].x, chb.channels[i].y, 'k')
            if i == 0:
                maxX = np.max(channels[i].x)
                minX = np.min(channels[i].x)
                maxY = np.max(channels[i].y)
                minY = np.min(channels[i].y)
            else:
                maxX = max(maxX, np.max(channels[i].x))
                minX = min(minX, np.min(channels[i].x))
                maxY = max(maxY, np.max(channels[i].y))
                minY = min(minY, np.min(channels[i].y))
        plt.axis([minX, maxX, minY-10*w, maxY+10*w])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        pts = np.zeros((2,2))
        for i in range(0,2):
            pt = np.asarray(plt.ginput(1))
            pts[i,:] = pt
            plt.scatter(pt[0][0], pt[0][1])
        plt.plot([pts[0,0], pts[1,0], pts[1,0], pts[0,0], pts[0,0]], [pts[0,1], pts[0,1], pts[1,1], pts[1,1], pts[0,1]], 'r')
        xmin = min(pts[0,0], pts[1,0])
        xmax = max(pts[0,0], pts[1,0])
        ymin = min(pts[0,1], pts[1,1])
        ymax = max(pts[0,1], pts[1,1])
    iwidth = int((xmax-xmin)/dx)
    iheight = int((ymax-ymin)/dx)
    topo = np.zeros((iheight, iwidth, 3*n_steps)) # array for storing topographic surfaces
    dists = np.zeros((iheight, iwidth, n_steps))
    zmaps = np.zeros((iheight, iwidth, n_steps))
    facies = np.zeros((3*n_steps, 1))
    # create initial topography:
    x1 = np.linspace(0, iwidth-1, iwidth)
    y1 = np.linspace(0, iheight-1, iheight)
    xv, yv = np.meshgrid(x1,y1)
    z1 = channels[0].z
    z1 = z1[(channels[0].x > xmin) & (channels[0].x < xmax)]
    topoinit = z1[0] - ((z1[0] - z1[-1]) / (xmax - xmin)) * xv * dx # initial (sloped) topography
    topo[:,:,0] = topoinit.copy()
    surf = topoinit.copy()
    facies[0] = np.nan
    # generate surfaces:
    channels3D = []
    for i in trange(n_steps):
        x = channels[i].x
        y = channels[i].y
        z = channels[i].z
        cutoff_ind = []
        # check if there were cutoffs during the last time step and collect indices in an array:
        for j in range(len(cot)):
            if (cot[j] >= sclt[i-1]) & (cot[j] < sclt[i]):
                cutoff_ind.append(j)
        # create distance map:
        cl_dist, x_pix, y_pix, z_pix, s_pix, z_map, x1, y1, z1 = dist_map(x, y, z, xmin, xmax, ymin, ymax, dx, delta_s)
        # erosion:
        surf = np.minimum(surf, erosion_surface(h,w/dx,cl_dist,z_map))
        topo[:,:,3*i] = surf # erosional surface
        dists[:,:,i] = cl_dist # distance map
        zmaps[:,:,i] = z_map # map of closest channel elevation
        facies[3*i] = np.nan # array for facies code

        if model_type == 'fluvial':
            z_map = gaussian_filter(z_map, sigma=50) # smooth z_map to avoid artefacts in levees
            pb = point_bar_surface(cl_dist, z_map, h, w/dx)
            th = np.maximum(surf,pb)-surf
            th[cl_dist > 1.0 * w/dx] = 0 # eliminate sand outside of channel
            th[th<0] = 0 # eliminate negative thickness values
            surf = surf+th # update topographic surface with sand thickness
            topo[:,:,3*i+1] = surf # top of sand
            facies[3*i+1] = 1 # facies code for point bar sand
            E_max = z_map + h_mud[i]
            if i == n_steps-1:
                surf_diff = E_max-surf
                surf_diff[surf_diff < 0] = 0
                plt.figure()
                plt.imshow(surf_diff)
            levee = fluvial_levee(cl_dist, surf, E_max, w/dx, diff_scale, v_fine, v_coarse, dt)
            if i == n_steps-1:
                plt.figure()
                plt.imshow(levee)
            surf = surf + levee # mud/levee deposition 
            topo[:,:,3*i+2] = surf # top of levee
            facies[3*i+2] = 2 # facies code for overbank
            channels3D.append(Channel(x1-xmin, y1-ymin, z1, w, h))

        if model_type == 'submarine':
            z_map = gaussian_filter(z_map, sigma=50) # smooth z_map to avoid artefacts in levees
            th, relief = sand_surface(surf, bth, dcr, z_map, h) # sandy channel deposit
            th[th < 0] = 0 # eliminate negative thickness values
            ws = w * (dcr/h)**0.5 # channel width at the top of the channel deposit
            th[cl_dist > 1.0 * ws/dx] = 0 # eliminate sand outside of channel
            surf = surf+th # update topographic surface with sand thickness
            topo[:,:,3*i+1] = surf # top of sand
            facies[3*i+1] = 1 # facies code for channel sand
            # need to blur z-map so that levees don't have artefacts:
            # blurred = filters.gaussian(z_map, sigma=(50, 50), truncate=3.5, multichannel=False)
            E_max = z_map + h_mud[i]
            levee = submarine_levee(h_mud[i], cl_dist, surf, E_max, w/dx, diff_scale, v_fine, v_coarse, dt)
            surf = surf + levee # mud/levee deposition
            topo[:,:,3*i+2] = surf # top of levee
            facies[3*i+2] = 2 # facies code for overbank 
            channels3D.append(Channel(x1-xmin, y1-ymin, z1, w, h))

    topo = np.concatenate((np.reshape(topoinit,(iheight,iwidth,1)),topo),axis=2) # add initial topography to array
    strat = topostrat(topo) # create stratigraphic surfaces
    strat = np.delete(strat, np.arange(3*n_steps+1)[1::3], 2) # get rid of unnecessary stratigraphic surfaces (duplicates)
    facies = np.delete(facies, np.arange(3*n_steps)[::3]) # get rid of unnecessary facies layers (NaNs)
    if model_type == 'fluvial':
        facies_code = {1:'point bar', 2:'levee'}
    if model_type == 'submarine':
        facies_code = {1:'channel sand', 2:'levee'}
    chb_3d = ChannelBelt3D(model_type, topo, strat, facies, facies_code, dx, channels3D)
    return chb_3d, xmin, xmax, ymin, ymax, dists, zmaps

def resample_centerline(x,y,z,deltas):
    """
    Resample centerline so that 'deltas' is roughly constant, using parametric 
    spline representation of curve; note that there is *no* smoothing.

    Parameters
    ----------
    x : array_like
        x-coordinates of centerline.
    y : array_like
        y-coordinates of centerline.
    z : array_like
        z-coordinates of centerline.
    deltas : float
        Distance between points on centerline.

    Returns
    -------
    x : ndarray
        x-coordinates of resampled centerline.
    y : ndarray
        y-coordinates of resampled centerline.
    z : ndarray
        z-coordinates of resampled centerline.
    dx : ndarray
        dx of resampled centerline.
    dy : ndarray
        dy of resampled centerline.
    dz : ndarray
        dz of resampled centerline.
    ds : ndarray
        ds of resampled centerline.
    s : ndarray
        s-coordinates of resampled centerline.
    """

    dx, dy, dz, ds, s = compute_derivatives(x,y,z) # compute derivatives
    tck, u = scipy.interpolate.splprep([x,y,z],s=0) 
    unew = np.linspace(0,1,1+int(round(s[-1]/deltas))) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    x, y, z = out[0], out[1], out[2] # assign new coordinate values
    dx, dy, dz, ds, s = compute_derivatives(x,y,z) # recompute derivatives
    return x,y,z,dx,dy,dz,ds,s

def migrate_one_step(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega=-1.0,gamma=2.5):
    """
    Migrate centerline during one time step, using the migration computed as in Howard & Knutson (1984).

    Parameters
    ----------
    x : array_like
        x-coordinates of centerline.
    y : array_like
        y-coordinates of centerline.
    z : array_like
        z-coordinates of centerline.
    W : float
        Channel width.
    kl : float
        Migration rate (or erodibility) constant (m/s).
    dt : float
        Duration of time step (s).
    k : float
        Constant for calculating the exponent alpha (= 1.0).
    Cf : float
        Dimensionless Chezy friction factor.
    D : float
        Channel depth.
    pad : int
        Padding parameter for migration rate computation.
    pad1 : int
        Padding parameter for centerline adjustment.
    omega : float
        Constant in Howard & Knutson equation (= -1.0).
    gamma : float
        Constant in Howard & Knutson equation (= 2.5).

    Returns
    -------
    x : array_like
        New x-coordinates of centerline after migration.
    y : array_like
        New y-coordinates of centerline after migration.
    """

    ns=len(x)
    curv = compute_curvature(x,y)
    dx, dy, dz, ds, s = compute_derivatives(x,y,z)
    # sinuosity = s[-1]/(x[-1]-x[0])
    sinuosity = s[-1]/(np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2))
    curv = W*curv # dimensionless curvature
    R0 = kl*curv # simple linear relationship between curvature and nominal migration rate
    alpha = k*2*Cf/D # exponent for convolution function G
    R1 = compute_migration_rate(pad,ns,ds,alpha,R0)
    R1 = sinuosity**(-2/3.0)*R1
    # calculate new centerline coordinates:
    dy_ds = dy[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    dx_ds = dx[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    # adjust x and y coordinates (this *is* the migration):
    x[pad1:ns-pad+1] = x[pad1:ns-pad+1] + R1[pad1:ns-pad+1]*dy_ds*dt  
    y[pad1:ns-pad+1] = y[pad1:ns-pad+1] - R1[pad1:ns-pad+1]*dx_ds*dt 
    return x,y

def migrate_one_step_w_bias(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega=-1.0,gamma=2.5):
    ns=len(x)
    curv = compute_curvature(x,y)
    dx, dy, dz, ds, s = compute_derivatives(x,y,z)
    sinuosity = s[-1]/(x[-1]-x[0])
    curv = W*curv # dimensionless curvature
    R0 = kl*curv # simple linear relationship between curvature and nominal migration rate
    alpha = k*2*Cf/D # exponent for convolution function G
    R1 = compute_migration_rate(pad,ns,ds,alpha,R0)
    R1 = sinuosity**(-2/3.0)*R1
    pad = -1
    # calculate new centerline coordinates:
    dy_ds = dy[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    dx_ds = dx[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    tilt_factor = 0.2
    T = kl*tilt_factor*np.ones(np.shape(x))
    angle = 90.0
    # adjust x and y coordinates (this *is* the migration):
    x[pad1:ns-pad+1] = x[pad1:ns-pad+1] + R1[pad1:ns-pad+1] * dy_ds * dt + T[pad1:ns-pad+1] * dy_ds * dt * (np.sin(np.deg2rad(angle)) * dx_ds + np.cos(np.deg2rad(angle)) * dy_ds)
    y[pad1:ns-pad+1] = y[pad1:ns-pad+1] - R1[pad1:ns-pad+1] * dx_ds * dt - T[pad1:ns-pad+1] * dx_ds * dt * (np.sin(np.deg2rad(angle)) * dx_ds + np.cos(np.deg2rad(angle)) * dy_ds)
    return x,y

def generate_initial_channel(W,D,Sl,deltas,pad,n_bends):
    """
    Generate a straight Channel object with some noise added that can serve
    as input for initializing a ChannelBelt object.

    Parameters
    ----------
    W : float
        Channel width.
    D : float
        Channel depth.
    Sl : float
        Channel gradient.
    deltas : float
        Distance between nodes on centerline.
    pad : int
        Padding (number of node points along centerline).
    n_bends : int
        Approximate number of bends to be simulated.

    Returns
    -------
    Channel
        A Channel object initialized with the generated coordinates and dimensions.
    """
    noisy_len = n_bends*10*W/2.0 # length of noisy part of initial centerline
    pad1 = int(pad/10.0) # padding at upstream end can be shorter than padding on downstream end
    if pad1<5:
        pad1 = 5
    x = np.linspace(0, noisy_len+(pad+pad1)*deltas, int(noisy_len/deltas+pad+pad1)+1) # x coordinate
    y = 2.0 * (2*np.random.random_sample(int(noisy_len/deltas)+1,)-1)
    y = np.hstack((np.zeros((pad1),),y,np.zeros((pad),))) # y coordinate
    deltaz = Sl * deltas*(len(x)-1)
    z = np.linspace(0,deltaz,len(x))[::-1] # z coordinate
    return Channel(x,y,z,W,D)

@numba.jit(nopython=True) # use Numba to speed up the heaviest computation
def compute_migration_rate(pad, ns, ds, alpha, R0, omega=-1.0, gamma=2.5):
    """
    Compute migration rate as weighted sum of upstream curvatures.

    Parameters
    ----------
    pad : int
        Padding (number of nodepoints along centerline).
    ns : int
        Number of points in centerline.
    ds : ndarray
        Distances between points in centerline.
    alpha : float
        Exponent for convolution function G.
    R0 : ndarray
        Nominal migration rate (dimensionless curvature * migration rate constant).
    omega : float, optional
        Constant in HK model, by default -1.0.
    gamma : float, optional
        Constant in HK model, by default 2.5.

    Returns
    -------
    ndarray
        Adjusted channel migration rate.
    """
    R1 = np.zeros(ns)  # preallocate adjusted channel migration rate
    pad1 = int(pad / 10.0)  # padding at upstream end can be shorter than padding on downstream end
    if pad1 < 5:
        pad1 = 5
    for i in range(pad1, ns - pad):
        si2 = np.hstack((np.array([0]), np.cumsum(ds[i - 1::-1])))  # distance along centerline, backwards from current point
        G = np.exp(-alpha * si2)  # convolution vector
        R1[i] = omega * R0[i] + gamma * np.sum(R0[i::-1] * G) / np.sum(G)  # main equation
    return R1

def compute_derivatives(x,y,z):
    """
    Compute first derivatives of a curve (centerline).

    Parameters
    ----------
    x : array_like
        Cartesian x-coordinates of the curve.
    y : array_like
        Cartesian y-coordinates of the curve.
    z : array_like
        Cartesian z-coordinates of the curve.

    Returns
    -------
    dx : ndarray
        First derivative of x coordinate.
    dy : ndarray
        First derivative of y coordinate.
    dz : ndarray
        First derivative of z coordinate.
    ds : ndarray
        Distances between consecutive points along the curve.
    s : ndarray
        Cumulative distance along the curve.
    """
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)   
    dz = np.gradient(z)   
    ds = np.sqrt(dx**2+dy**2+dz**2)
    s = np.hstack((0,np.cumsum(ds[1:])))
    return dx, dy, dz, ds, s

def compute_curvature(x,y):
    """
    Compute the first derivatives and curvature of a curve (centerline).

    Parameters
    ----------
    x : array_like
        Cartesian coordinates of the curve along the x-axis.
    y : array_like
        Cartesian coordinates of the curve along the y-axis.

    Returns
    -------
    curvature : ndarray
        Curvature of the curve (in 1/units of x and y).

    Notes
    -----
    The function calculates the first and second derivatives of the input coordinates
    and uses them to compute the curvature of the curve.
    """
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)      
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature

def make_colormap(seq):
    """
    Return a LinearSegmentedColormap.

    Parameters
    ----------
    seq : list of tuple
        A sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0, 1).

    Returns
    -------
    LinearSegmentedColormap
        A colormap object that can be used in matplotlib plotting.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
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
    """
    Find the indices of the k-th diagonal of a 2D array.

    Parameters
    ----------
    a : array_like
        Input array. Must be 2-dimensional.
    k : int
        Diagonal offset. If k=0, the main diagonal is returned. 
        If k>0, the k-th upper diagonal is returned. 
        If k<0, the k-th lower diagonal is returned.

    Returns
    -------
    tuple of ndarray
        A tuple of arrays (rows, cols) containing the row and column indices 
        of the k-th diagonal.

    Notes
    -----
    This function is adapted from a solution on Stack Overflow:
    https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices
    """
    rows, cols = np.diag_indices_from(a)
    if k<0:
        return rows[:k], cols[-k:]
    elif k>0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols
    
def find_cutoffs(x,y,crdist,deltas):
    """
    Identify locations of cutoffs along a centerline and the indices of the segments 
    that will become part of the oxbows.

    Parameters
    ----------
    x : array_like
        x-coordinates of the centerline.
    y : array_like
        y-coordinates of the centerline.
    crdist : float
        Critical cutoff distance.
    deltas : float
        Distance between neighboring points along the centerline.

    Returns
    -------
    ind1 : ndarray
        Indices of the first set of cutoff points.
    ind2 : ndarray
        Indices of the second set of cutoff points.
    """
    diag_blank_width = int((crdist+20*deltas)/deltas)
    # distance matrix for centerline points:
    dist = distance.cdist(np.array([x,y]).T,np.array([x,y]).T)
    dist[dist>crdist] = np.nan # set all values that are larger than the cutoff threshold to NaN
    # set matrix to NaN along the diagonal zone:
    for k in range(-diag_blank_width,diag_blank_width+1):
        rows, cols = kth_diag_indices(dist,k)
        dist[rows,cols] = np.nan
    i1, i2 = np.where(~np.isnan(dist))
    ind1 = i1[np.where(i1<i2)[0]] # get rid of unnecessary indices
    ind2 = i2[np.where(i1<i2)[0]] # get rid of unnecessary indices
    return ind1, ind2 # return indices of cutoff points and cutoff coordinates

def cut_off_cutoffs(x,y,z,s,crdist,deltas):
    """
    Execute cutoffs by removing oxbows from the centerline and storing cutoff coordinates.

    Parameters
    ----------
    x : array_like
        x-coordinates of the centerline.
    y : array_like
        y-coordinates of the centerline.
    z : array_like
        z-coordinates of the centerline.
    s : array_like
        Additional parameter (not used in the function).
    crdist : float
        Critical cutoff distance.
    deltas : array_like
        Distance between neighboring points along the centerline.

    Returns
    -------
    x : array_like
        Updated x-coordinates of the centerline after cutoffs.
    y : array_like
        Updated y-coordinates of the centerline after cutoffs.
    z : array_like
        Updated z-coordinates of the centerline after cutoffs.
    xc : list of array_like
        Lists of x-coordinates of cutoff segments.
    yc : list of array_like
        Lists of y-coordinates of cutoff segments.
    zc : list of array_like
        Lists of z-coordinates of cutoff segments.
    """
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
    """
    Find coordinates of channel banks, given a centerline and a channel width.

    Parameters
    ----------
    x : array_like
        x-coordinates of the centerline.
    y : array_like
        y-coordinates of the centerline.
    W : float
        Channel width.

    Returns
    -------
    xm : ndarray
        x-coordinates of the channel banks (both left and right banks).
    ym : ndarray
        y-coordinates of the channel banks (both left and right banks).
    """
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

def dist_map(x, y, z, xmin, xmax, ymin, ymax, dx, delta_s):
    """
    Function for centerline rasterization and distance map calculation.

    Parameters
    ----------
    x : array_like
        x coordinates of centerline.
    y : array_like
        y coordinates of centerline.
    z : array_like
        z coordinates of centerline.
    xmin : float
        Minimum x coordinate that defines the area of interest.
    xmax : float
        Maximum x coordinate that defines the area of interest.
    ymin : float
        Minimum y coordinate that defines the area of interest.
    ymax : float
        Maximum y coordinate that defines the area of interest.
    dx : float
        Grid cell size (m).
    delta_s : float
        Distance between points along centerline (m).

    Returns
    -------
    cl_dist : ndarray
        Distance map (distance from centerline).
    x_pix : ndarray
        x pixel coordinates of the centerline.
    y_pix : ndarray
        y pixel coordinates of the centerline.
    z_pix : ndarray
        z pixel coordinates of the centerline.
    s_pix : ndarray
        Along-channel distance in pixels.
    z_map : ndarray
        Map of reference channel thalweg elevation (elevation of closest point along centerline).
    x : ndarray
        x centerline coordinates clipped to the 3D model domain.
    y : ndarray
        y centerline coordinates clipped to the 3D model domain.
    z : ndarray
        z centerline coordinates clipped to the 3D model domain.
    """
    y = y[(x>xmin) & (x<xmax)]
    z = z[(x>xmin) & (x<xmax)]
    x = x[(x>xmin) & (x<xmax)]
    x = x[(y>ymin) & (y<ymax)]
    z = z[(y>ymin) & (y<ymax)]
    y = y[(y>ymin) & (y<ymax)]
    dummy,dy,dz,ds,s = compute_derivatives(x,y,z)
    if len(np.where(ds>2*delta_s)[0])>0:
        inds = np.where(ds>2*delta_s)[0]
        inds = np.hstack((0,inds,len(x)))
        lengths = np.diff(inds)
        long_segment = np.where(lengths==max(lengths))[0][0]
        start_ind = inds[long_segment]+1
        end_ind = inds[long_segment+1]
        if end_ind<len(x):
            x = x[start_ind:end_ind]
            y = y[start_ind:end_ind]
            z = z[start_ind:end_ind] 
        else:
            x = x[start_ind:]
            y = y[start_ind:]
            z = z[start_ind:]
    xdist = xmax - xmin
    ydist = ymax - ymin
    iwidth = int((xmax-xmin)/dx)
    iheight = int((ymax-ymin)/dx)
    xratio = iwidth/xdist
    # create list with pixel coordinates:
    pixels = []
    for i in range(0,len(x)):
        px = int(iwidth - (xmax - x[i]) * xratio)
        py = int(iheight - (ymax - y[i]) * xratio)
        pixels.append((px,py))
    # create image and numpy array:
    img = Image.new("RGB", (iwidth, iheight), "white")
    draw = ImageDraw.Draw(img)
    draw.line(pixels, fill="rgb(0, 0, 0)") # draw centerline as black line
    pix = np.array(img)
    cl = pix[:,:,0]
    cl[cl==255] = 1 # set background to 1 (centerline is 0)
    y_pix,x_pix = np.where(cl==0) 
    x_pix,y_pix = order_cl_pixels(x_pix, y_pix, img)
    # This next block of code is kind of a hack. Looking for, and eliminating, 'bad' pixels.
    img = np.array(img)
    img = img[:,:,0]
    img[img==255] = 1 
    img1 = morphology.binary_dilation(img, morphology.square(2)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img, img1)
        x_pix,y_pix = order_cl_pixels(x_pix, y_pix, img) 
    img1 = morphology.binary_dilation(img, np.array([[1,0,1], [1,1,1]], dtype=np.uint8)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix, y_pix, img)
    img1 = morphology.binary_dilation(img, np.array([[1,0,1], [0,1,0], [1,0,1]], dtype=np.uint8)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img, img1)
        x_pix,y_pix = order_cl_pixels(x_pix, y_pix, img)
    #redo the distance calculation (because x_pix and y_pix do not always contain all the points in cl):
    cl[cl==0] = 1
    cl[y_pix,x_pix] = 0
    cl_dist, inds = ndimage.distance_transform_edt(cl, return_indices=True)
    dx,dy,dz,ds,s = compute_derivatives(x,y,z)
    dx_pix = np.diff(x_pix)
    dy_pix = np.diff(y_pix)
    ds_pix = np.sqrt(dx_pix**2 + dy_pix**2)
    s_pix = np.hstack((0,np.cumsum(ds_pix)))
    f = scipy.interpolate.interp1d(s, z)
    snew = s_pix*s[-1]/s_pix[-1]
    if snew[-1] > s[-1]:
        snew[-1] = s[-1]
    snew[snew<s[0]] = s[0]
    z_pix = f(snew)
    # create z_map:
    z_map = np.zeros(np.shape(cl_dist)) 
    z_map[y_pix, x_pix] = z_pix
    xinds = inds[1,:,:]
    yinds = inds[0,:,:]
    for i in range(0, len(x_pix)):
        z_map[(xinds==x_pix[i]) & (yinds==y_pix[i])] = z_pix[i]
    return cl_dist, x_pix, y_pix, z_pix, s_pix, z_map, x, y, z

def erosion_surface(h,w,cl_dist,z):
    """
    Function for creating a parabolic erosional surface.

    Parameters
    ----------
    h : float
        Geomorphic channel depth (m).
    w : int
        Geomorphic channel width (in pixels, as cl_dist is also given in pixels).
    cl_dist : numpy.ndarray
        Distance map (distance from centerline).
    z : float
        Reference elevation (m).

    Returns
    -------
    surf : numpy.ndarray
        Map of the erosional surface (m).
    """
    surf = z + (4*h/w**2)*(cl_dist+w*0.5)*(cl_dist-w*0.5)
    return surf

def point_bar_surface(cl_dist,z,h,w):
    """
    Create a Gaussian-based point bar surface used in a 3D fluvial model.

    Parameters
    ----------
    cl_dist : array-like
        Distance map (distance from centerline) in pixels.
    z : float
        Reference elevation in meters.
    h : float
        Channel depth in meters.
    w : float
        Channel width in pixels.

    Returns
    -------
    pb : array-like
        Map of the Gaussian surface that can be used to form a point bar deposit in meters.
    """
    pb = z-h*np.exp(-(cl_dist**2)/(2*(w*0.33)**2))
    return pb

def sand_surface(surf, bth, dcr, z_map, h):
    """
    Function for creating the top horizontal surface sand-rich deposit in the bottom of the channel
    used in 3D submarine channel models.

    Parameters
    ----------
    surf : ndarray
        Current geomorphic surface.
    bth : float
        Thickness of sand deposit in axis of channel (m).
    dcr : float
        Critical channel depth, above which there is no sand deposition (m).
    z_map : ndarray
        Map of reference channel thalweg elevation (elevation of closest point along centerline).
    h : float
        Channel depth (m).

    Returns
    -------
    th : ndarray
        Thickness map of sand deposit (m).
    relief : ndarray
        Map of channel relief (m).
    """
    relief = np.abs(surf - z_map + h)
    relief = np.abs(relief - np.amin(relief))
    th = bth * (1 - relief/dcr) # bed thickness inversely related to relief
    th[th<0] = 0.0 # set negative th values to zero
    return th, relief

def fluvial_levee(cl_dist, topo, E_max, w, diff_scale, v_fine, v_coarse, dt):
    """
    Function for creating a levee layer in a fluvial 3D model based on the diffusion-based overbank deposition model of Howard, 1992.

    Parameters
    ----------
    cl_dist : ndarray
        Distance map (distance from centerline).
    topo : ndarray
        Current topographic surface.
    E_max : float
        Maximum thickness of overbank deposit (specific to location).
    w : float
        Channel width.
    diff_scale : float
        Diffusion length scale.
    v_fine : float
        Deposition rate of fine sediment, in m/year (for overbank deposition).
    v_coarse : float
        Deposition rate of coarse sediment, in m/year (for overbank deposition).
    dt : float
        Time step (in seconds).

    Returns
    -------
    levee : ndarray
        Thickness of levee layer (same size as 'topo').
    """
    dep_rate = (E_max - topo) * (v_fine + v_coarse * np.exp(-cl_dist/diff_scale))
    dep_rate[cl_dist < 0.6*w] = 0  # get rid of the mud in the active channel
    dep_rate[dep_rate < 0] = 0
    levee = dep_rate * (dt/(365*24*60*60))
    return levee

def submarine_levee(h_mud, cl_dist, topo, E_max, w, diff_scale, v_fine, v_coarse, dt):
    """
    Function for creating a levee layer in a submarine 3D model based on the diffusion-based overbank deposition model of Howard, 1992.

    Parameters
    ----------
    h_mud : float
        Maximum thickness of overbank deposit (specific to time step).
    cl_dist : ndarray
        Distance map (distance from centerline).
    topo : ndarray
        Current topographic surface.
    E_max : float
        Maximum thickness of overbank deposit (specific to location).
    w : float
        Channel width.
    diff_scale : float
        Diffusion length scale.
    v_fine : float
        Deposition rate of fine sediment, in m/year (for overbank deposition).
    v_coarse : float
        Deposition rate of coarse sediment, in m/year (for overbank deposition).
    dt : float
        Time step (in seconds).

    Returns
    -------
    levee : ndarray
        Thickness of levee layer (same size as 'topo').
    """
    dep_rate = (E_max - topo) * (v_fine + v_coarse * np.exp(-cl_dist/diff_scale))
    dep_rate[dep_rate < 0] = 0
    levee = dep_rate * (dt/(365*24*60*60))
    surf3 = h_mud + (4*h_mud / w**2) * (cl_dist + w*0.5) * (cl_dist - w*0.5) # 'erosional' surface
    levee = np.minimum(levee, surf3) # get rid of the mud in the axis of the active channel
    return levee

def topostrat(topo):
    """
    Convert a stack of geomorphic surfaces into stratigraphic surfaces.

    Parameters
    ----------
    topo : numpy.ndarray
        3D numpy array of geomorphic surfaces, with the oldest at index 0.

    Returns
    -------
    strat : numpy.ndarray
        3D numpy array of stratigraphic surfaces, with the oldest at index 0.

    Notes
    -----
    The assumption is that the topographic array has the oldest surface at the '0' z (third) index and therefore needs to be flipped twice.
    """
    strat = np.minimum.accumulate(topo[:, :, ::-1], axis=2)[:, :, ::-1] # this eliminates the 'for' loop and is therefore faster
    return strat

def eliminate_bad_pixels(img, img1):
    """
    Function for removing 'bad' pixels along channel centerline.

    Parameters
    ----------
    img : ndarray
        Black-and-white image of channel centerline (centerline pixels are 0).
    img1 : ndarray
        Dilated version of centerline image 'img'.

    Returns
    -------
    x_pix : ndarray
        Cleaned array of x pixel coordinates.
    y_pix : ndarray
        Cleaned array of y pixel coordinates.
    """
    x_ind = np.where(img1==0)[1][0]
    y_ind = np.where(img1==0)[0][0]
    img[y_ind:y_ind+2,x_ind:x_ind+2] = np.ones(1,).astype(np.uint8)
    all_labels = measure.label(img,background=1,connectivity=2)
    cl=all_labels.copy()
    cl[cl==2]=0
    cl[cl>0]=1
    y_pix,x_pix = np.where(cl==1)
    return x_pix, y_pix

def order_cl_pixels(x_pix, y_pix, img):
    """
    Function for ordering pixels along a channel centerline, starting on the left side.

    Parameters
    ----------
    x_pix : array_like
        Unordered x pixel coordinates of the centerline.
    y_pix : array_like
        Unordered y pixel coordinates of the centerline.
    img : array_like
        Image array used to determine the distances from the edges.

    Returns
    -------
    x_pix : array_like
        Ordered x pixel coordinates of the centerline.
    y_pix : array_like
        Ordered y pixel coordinates of the centerline.
    """
    dist = distance.cdist(np.array([x_pix,y_pix]).T,np.array([x_pix,y_pix]).T)
    dist[np.diag_indices_from(dist)]=100.0
    # ind = np.argmin(x_pix) # select starting point on left side of image
    x_pix_dist_from_edges = np.shape(img)[1] - max(x_pix) + min(x_pix)
    y_pix_dist_from_edges = np.shape(img)[0] - max(y_pix) + min(y_pix)
    if x_pix_dist_from_edges <= y_pix_dist_from_edges:
        ind = np.argmin(x_pix) # select starting point on left side of image
    else:
        ind = np.argmin(y_pix) # select starting point on lower side of image 
    clinds = [ind]
    count = 0
    while count<len(x_pix):
        t = dist[ind,:].copy()
        if len(clinds)>2:
            t[clinds[-2]]=t[clinds[-2]]+100.0
            t[clinds[-3]]=t[clinds[-3]]+100.0
        ind = np.argmin(t)
        clinds.append(ind)
        count=count+1
    x_pix = x_pix[clinds]
    y_pix = y_pix[clinds]
    return x_pix,y_pix

def save_3d_chb_to_hdf5(chb_3d, fname):
    """
    Save a 3D channelbelt model as an HDF5 file.

    Parameters
    ----------
    chb_3d : ChannelBelt3D
        The ChannelBelt3D object to be saved.
    fname : str
        The filename for the HDF5 file.
    """
    f = h5py.File(fname,'w')
    grp = f.create_group('model')
    grp.create_dataset('dx', data = chb_3d.dx)
    grp.create_dataset('topo', data = chb_3d.topo)
    grp.create_dataset('strat', data = chb_3d.strat)
    grp.create_dataset('facies', data = chb_3d.facies)
    for key in chb_3d.facies_code.keys():
        grp.create_dataset(chb_3d.facies_code[key], data = key)
    grp = f.create_group('channels')
    depths = []; widths = []; xcoords = []; ycoords = []; zcoords = []; lengths = []
    for channel in chb_3d.channels:
        depths.append(channel.D)
        widths.append(channel.W)
        xcoords.append(channel.x)
        ycoords.append(channel.y)
        zcoords.append(channel.z)
        lengths.append(len(channel.x))
    x = np.nan * np.ones((len(xcoords), max(lengths)))
    for i in range(len(xcoords)):
        x[i, :len(xcoords[i])] = xcoords[i]
    y = np.nan * np.ones((len(ycoords), max(lengths)))
    for i in range(len(ycoords)):
        y[i, :len(ycoords[i])] = ycoords[i]
    z = np.nan * np.ones((len(zcoords), max(lengths)))
    for i in range(len(zcoords)):
        z[i, :len(zcoords[i])] = zcoords[i]
    grp.create_dataset('depths', data = depths)    
    grp.create_dataset('widths', data = widths)    
    grp.create_dataset('x', data = x)
    grp.create_dataset('y', data = y)
    grp.create_dataset('z', data = z)
    f.close()

def read_3d_chb_from_hdf5(model_type, fname):
    """
    Function for reading 3D channelbelt model from an HDF5 file (that was saved using 'save_3d_chb_to_hdf5').

    Parameters
    ----------
    model_type : str
        Model type (can be 'fluvial' or 'submarine').
    fname : str
        Filename of the HDF5 file.

    Returns
    -------
    ChannelBelt3D
        ChannelBelt3D object that was created from the HDF5 file.
    """
    f = h5py.File(fname, 'r')
    model  = f['model']
    topo = np.array(model['topo'])
    strat = np.array(model['strat'])
    facies = np.array(model['facies'])
    facies_code = {}
    facies_code[int(np.array(model['point bar']))] = 'point bar'
    facies_code[int(np.array(model['levee']))] = 'levee'
    dx = float(np.array(model['dx']))
    x = np.array(f['channels']['x'])
    y = np.array(f['channels']['y'])
    z = np.array(f['channels']['z'])
    depths = np.array(f['channels']['depths'])
    widths = np.array(f['channels']['widths'])
    channels = []
    for i in range(x.shape[0]):
        x1 = x[i, :]
        x1 = x1[np.isnan(x1) == 0]
        y1 = y[i, :]
        y1 = y1[np.isnan(y1) == 0]
        z1 = z[i, :]
        z1 = z1[np.isnan(z1) == 0]
        channels.append(Channel(x1, y1, z1, widths[i], depths[i]))
    chb_3d = ChannelBelt3D(model_type, topo, strat, facies, facies_code, dx, channels)
    f.close()
    return chb_3d

def create_poro_perm(chb_3d, poro_max):
    """
    Generate porosity and permeability fields from 3D channelbelt model.

    Parameters
    ----------
    chb_3d : ChannelBelt3D
        A ChannelBelt3D object with these attributes:
        - strat: 3D numpy array representing stratigraphy.
        - topo: 3D numpy array representing topography.
    poro_max : float
        Maximum porosity value.

    Returns
    -------
    porosity : numpy.ndarray
        3D numpy array of porosity values.
    permeability : numpy.ndarray
        3D numpy array of permeability values.

    Notes
    -----
    The function calculates porosity as a function of height above the thalweg (HAT) and assigns
    permeability based on the calculated porosity. Areas with zero thickness are assigned zero porosity.
    """
    ny, nx, nz = np.shape(chb_3d.strat)
    porosity = np.zeros((ny-1, nx-1, nz - 1))
    for i in range(int((chb_3d.strat.shape[2] - 1)/2)): # only working with channel sands
        hat = np.abs(chb_3d.topo[:, :, 3*i + 1] - np.min(chb_3d.topo[:, :, 3*i + 1])) # height above thalweg
        th = chb_3d.topo[:, :, 3*i + 2] - chb_3d.topo[:, :, 3*i + 1] # thickness of channel deposit
        hat[th == 0] = 100 # set a large HAT value where thickness is zero (we want zero porosity here)
        t = 0.25*(hat[1:,1:]+hat[1:,:-1]+hat[:-1,1:]+hat[:-1,:-1]) # average HAT for porosity grid
        t = t - np.min(t)
        t[t > 30.0] = 30.0
        # porosity is a function of elevation (the higher the elevation, the lower the porosity):
        t = poro_max - poro_max*(t/30.0)
        porosity[:, :, 2*i] = t # assign porosity
    porosity[porosity > poro_max] = poro_max
    permeability = 10**(17*porosity - 3)
    permeability[porosity == 0] = 0
    return porosity, permeability

def save_3d_chb_to_generic_hdf5(chb_3d, props, prop_names, fname):
    """
    Save a 3D channelbelt model as an HDF5 file.

    Parameters
    ----------
    chb_3d : ChannelBelt3D
        ChannelBelt3D object to be saved.
    props : list of ndarray
        List of property arrays (e.g., porosity, permeability).
    prop_names : list of str
        List of property names corresponding to the property arrays.
    fname : str
        Filename for the HDF5 file.

    Returns
    -------
    None
    """
    f = h5py.File(fname,'w')
    grp = f.create_group('model')
    grp.create_dataset('dx', data = chb_3d.dx)
    grp.create_dataset('topo', data = chb_3d.topo)
    grp.create_dataset('strat', data = chb_3d.strat)
    grp.create_dataset('facies', data = chb_3d.facies)
    count = 0
    for prop in props:
        grp.create_dataset(prop_names[count], data = prop)
        count += 1
    for key in chb_3d.facies_code.keys():
        grp.create_dataset(chb_3d.facies_code[key], data = key)
    f.close()

def write_eclipse_grid(strat, porosity, permeability, dx, fname):
    """
    Function for exporting an Eclipse file ('.grdecl' format) from an array of stratigraphic surfaces ('strat') and an array of porosity ('porosity').
    Additional 'keywords' like ACTNUM and SATNUM can be added the same way as porosity.

    Ordering of cornerpoints for first (uppermost) surface in Eclipse:
    ----------------------------------------
    | 1        2 | 3        4 | 5        6 | 
    |            |            |            |
    | 7        8 | 9       10 | 11      12 |
    ----------------------------------------
    | 13      14 | 15      16 | 17      18 |
    |            |            |            |
    | 19      20 | 21      22 | 23      24 |
    ----------------------------------------

    Parameters
    ----------
    strat : numpy.ndarray
        Stratigraphic surfaces (outputs from channel model).
    porosity : numpy.ndarray
        Porosity grid.
    dx : float
        Grid cell size (in meters).
    fname : str
        Filename of the Eclipse file to be written.
    """

    # these swaps have to be done because the logic below was written for this ordering of the axes
    surfs = np.swapaxes(strat, 0, 2) 
    surfs = np.swapaxes(surfs, 1, 2)
    porosity = np.swapaxes(porosity, 0, 2)
    porosity = np.swapaxes(porosity, 1, 2)
    permeability = np.swapaxes(permeability, 0, 2)
    permeability = np.swapaxes(permeability, 1, 2)
    
    nz,ny,nx = np.shape(surfs);
    nz=nz-1 # number of cells in z direction
    ny=ny-1 # number of cells in y direction
    nx=nx-1 # number of cells in x direction

    dy=dx  # size of cells in x and y directions

    print('creating cornerpoint array(ZCORN)')
    zcorn = np.zeros((8*nx*ny*nz,))
    for k in range(nz):
        # write cornerpoints for the top of layer 'k':
        surf = np.squeeze(surfs[k,:,:])
        zc = np.zeros((2*ny,2*nx))
        zc[::2,::2] = surf[:-1,:-1]
        zc[1::2,1::2] = surf[1:,1:]
        zc[1::2,::2] = surf[1:,:-1]
        zc[::2,1::2] = surf[:-1,1:]
        zc = np.reshape(zc,(1,4*nx*ny))
        zcorn[(2*k)*4*nx*ny : (2*k+1)*4*nx*ny] = zc;
        
        # write cornerpoints for the bottom of layer 'k':
        surf = np.squeeze(surfs[k+1,:,:])
        zc = np.zeros((2*ny,2*nx));
        zc[::2,::2] = surf[:-1,:-1]
        zc[1::2,1::2] = surf[1:,1:]
        zc[1::2,::2] = surf[1:,:-1]
        zc[::2,1::2] = surf[:-1,1:]
        zc = np.reshape(zc,(1,4*nx*ny))
        zcorn[(2*k+1)*4*nx*ny : (2*k+2)*4*nx*ny] = zc
    zcorn = np.reshape(zcorn, (int(len(zcorn)/8), 8))
    zcorn = 100 * zcorn # convert meters to centimeters

    print('creating pillar matrix (COORD)')
    coord = np.zeros(((nx+1)*(ny+1),6));
    for j in range(ny+1):
        for i in range(nx+1):
            coord[j*(nx+1)+i,0] = i*dx
            coord[j*(nx+1)+i,1] = j*dy
            coord[j*(nx+1)+i,2] = surfs[0,j,i] 
            coord[j*(nx+1)+i,3] = i*dx
            coord[j*(nx+1)+i,4] = j*dy
            coord[j*(nx+1)+i,5] = surfs[-1,j,i]
    coord = 100 * coord # convert meters to centimeters

    print('creating porosity array (PORO)')
    poro = np.zeros((nx*ny*nz,))
    [i,j,k] = np.meshgrid(np.arange(nx),np.arange(ny),np.arange(nz))
    ind1 = k*nx*ny + j*nx+i 
    ind2 = np.ravel_multi_index((k,j,i),(nz,ny,nx))
    poro[ind1] = porosity.flatten()[ind2]
    if np.mod(len(poro),8)==0: # if length of porosity array is a multiple of 8
        poro_1 = np.reshape(poro, (int(len(poro)/8), 8))
    else:
        poro_1 = np.reshape(poro[:-np.mod(len(poro),8)], (int(len(poro)/8), 8))

    print('creating permeability array (PERM)')
    perm = np.zeros((nx*ny*nz,))
    perm[ind1] = permeability.flatten()[ind2]
    if np.mod(len(perm),8)==0: # if length of porosity array is a multiple of 8
        perm_1 = np.reshape(perm, (int(len(perm)/8), 8))
    else:
        perm_1 = np.reshape(perm[:-np.mod(len(perm),8)], (int(len(perm)/8), 8))
    
    # write file:
    fid = open(fname, 'a')
    fid.write('SPECGRID\n')
    fid.write('%d %d %d' %(nx, ny, nz) + ' 1 F /\n')
    fid.write('COORD\n')
    print('writing pillars...')
    for i in range(coord.shape[0]):
        fid.write('%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n' %tuple(coord[i,:]))
    fid.write('/\n')
    fid.write(' ')
    fid.write('\n')
    fid.write('ZCORN\n')
    print('writing zcorns...')
    for i in range(zcorn.shape[0]):
        fid.write('%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n' %tuple(zcorn[i,:]))
    fid.write('/\n')
    fid.write(' ')
    fid.write('\n')
    fid.write('PORO\n')
    print('writing porosity...')
    for i in range(poro_1.shape[0]):
        fid.write('%6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\n' %tuple(poro_1[i,:]))
    if np.mod(len(poro),8)!=0: # if length of porosity array is not a multiple of 8
        for i in range(np.mod(len(poro),8)):
            fid.write('%6.4f ' %poro[-np.mod(len(poro),8):][i])
        fid.write('\n')
    fid.write('/\n')
    fid.write(' ')
    fid.write('\n')
    fid.write('PERMX\n')
    print('writing permeability...')
    for i in range(perm_1.shape[0]):
        fid.write('%6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f %6.4f\n' %tuple(perm_1[i,:]))
    if np.mod(len(perm),8)!=0: # if length of porosity array is not a multiple of 8
        for i in range(np.mod(len(perm),8)):
            fid.write('%6.4f ' %perm[-np.mod(len(perm),8):][i])
        fid.write('\n')
    fid.write('/\n')
    fid.close()
