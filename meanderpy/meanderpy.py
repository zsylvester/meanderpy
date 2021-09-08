import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.spatial import distance
from scipy import ndimage
from PIL import Image, ImageDraw
from skimage import measure
from skimage import morphology
from skimage import filters
from matplotlib.colors import LinearSegmentedColormap
import time, sys
import numba
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import trange

class Channel:
    """class for Channel objects"""

    def __init__(self,x,y,z,W,D):
        """Initialize Channel object

        :param x: x-coordinate of centerline
        :param y: y-coordinate of centerline
        :param z: z-coordinate of centerline
        :param W: channel width
        :param D: channel depth"""

        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D

class Cutoff:
    """class for Cutoff objects"""

    def __init__(self,x,y,z,W,D):
        """Initialize Cutoff object

        :param x: x-coordinate of centerline
        :param y: y-coordinate of centerline
        :param z: z-coordinate of centerline
        :param W: channel width
        :param D: channel depth"""

        self.x = x
        self.y = y
        self.z = z
        self.W = W
        self.D = D

class ChannelBelt3D:
    """class for 3D models of channel belts"""

    def __init__(self, model_type, topo, strat, facies, facies_code, dx, channels):
        """initialize ChannelBelt3D object

        :param model_type: type of model to be built; can be either 'fluvial' or 'submarine'
        :param topo: set of topographic surfaces (3D numpy array)
        :param strat: set of stratigraphic surfaces (3D numpy array)
        :param facies: facies volume (3D numpy array)
        :param facies_code: dictionary of facies codes, e.g. {0:'oxbow', 1:'point bar', 2:'levee'}
        :param dx: gridcell size (m)
        :param channels: list of channel objects that form 3D model"""

        self.model_type = model_type
        self.topo = topo
        self.strat = strat
        self.facies = facies
        self.facies_code = facies_code
        self.dx = dx
        self.channels = channels

    def plot_xsection(self, xsec, colors, ve):
        """method for plotting a cross section through a 3D model; also plots map of 
        basal erosional surface and map of final geomorphic surface

        :param xsec: location of cross section along the x-axis (in pixel/ voxel coordinates) 
        :param colors: list of RGB values that define the colors for different facies
        :param ve: vertical exaggeration
        :return: handles to the three figures"""

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
        """initialize ChannelBelt object

        :param channels: list of Channel objects
        :param cutoffs: list of Cutoff objects
        :param cl_times: list of ages of Channel objects (in years)
        :param cutoff_times: list of ages of Cutoff objects"""

        self.channels = channels
        self.cutoffs = cutoffs
        self.cl_times = cl_times
        self.cutoff_times = cutoff_times

    def migrate(self, nit, saved_ts, deltas, pad, crdist, depths, Cfs, kl, kv, dt, dens, t1, t2, t3, aggr_factor):
        """method for computing migration rates along channel centerlines and moving the centerlines accordingly

        :param nit: number of iterations
        :param saved_ts: which time steps will be saved; e.g., if saved_ts = 10, every tenth time step will be saved
        :param deltas: distance between nodes on centerline
        :param pad: padding (number of nodepoints along centerline)
        :param crdist: threshold distance at which cutoffs occur
        :param depths: array of channel depths (can very across iterations)
        :param Cf: array of dimensionless Chezy friction factors (can vary across iterations)
        :param kl: migration rate constant (m/s)
        :param kv: vertical slope-dependent erosion rate constant (m/s)
        :param dt: time step (s)
        :param dens: density of fluid (kg/m3)
        :param t1: time step when incision starts
        :param t2: time step when lateral migration starts
        :param t3: time step when aggradation starts
        :param aggr_factor: aggradation factor
        :param D: channel depth (m)"""

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
        omega = -1.0 # constant in migration rate calculation (Howard and Knutson, 1984)
        gamma = 2.5 # from Ikeda et al., 1981 and Howard and Knutson, 1984
        for itn in trange(nit): # main loop
            D = depths[itn]
            Cf = Cfs[itn]
            x, y = migrate_one_step(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma)
            # x, y = migrate_one_step_w_bias(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma)
            x,y,z,xc,yc,zc = cut_off_cutoffs(x,y,z,s,crdist,deltas) # find and execute cutoffs
            x,y,z,dx,dy,dz,ds,s = resample_centerline(x,y,z,deltas) # resample centerline
            slope = np.gradient(z)/ds
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
            if (np.mod(itn, saved_ts) == 0) & (itn > 0):
                self.cl_times.append(last_cl_time+(itn+1)*dt/(365*24*60*60.0))
                channel = Channel(x,y,z,W,D) # create channel object
                self.channels.append(channel)

    def plot(self, plot_type, pb_age, ob_age, end_time, n_channels):
        """method  for plotting ChannelBelt object

        :param plot_type: can be either 'strat' (for stratigraphic plot), 'morph' (for morphologic plot), or 'age' (for age plot)
        :param pb_age: age of point bars (in years) at which they get covered by vegetation
        :param ob_age: age of oxbow lakes (in years) at which they get covered by vegetation
        :param end_time: age of last channel to be plotted (in years)
        :param n_channels: total number of channels (used in 'age' plots; can be larger than number of channels being plotted)
        :return: handle to figure"""

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
            plt.fill(xm, ym, color=(16/255.0,73/255.0,90/255.0), zorder=order) #,edgecolor='k')
        plt.axis('equal')
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)
        return fig

    def create_movie(self, xmin, xmax, plot_type, filename, dirname, pb_age, ob_age, end_time, n_channels):
        """method for creating movie frames (PNG files) that capture the plan-view evolution of a channel belt through time
        movie has to be assembled from the PNG file after this method is applied

        :param xmin: value of x coodinate on the left side of frame
        :param xmax: value of x coordinate on right side of frame
        :param plot_type: plot type; can be either 'strat' (for stratigraphic plot) or 'morph' (for morphologic plot)
        :param filename: first few characters of the output filenames
        :param dirname: name of directory where output files should be written
        :param pb_age: age of point bars (in years) at which they get covered by vegetation (if the 'morph' option is used for 'plot_type')
        :param ob_age: age of oxbow lakes (in years) at which they get covered by vegetation (if the 'morph' option is used for 'plot_type')
        :param scale: scaling factor (e.g., 2) that determines how many times larger you want the frame to be, compared to the default scaling of the figure
        :param end_time: time at which simulation should be stopped
        :param n_channels: total number of channels + cutoffs for which simulation is run (usually it is len(chb.cutoffs) + len(chb.channels)). Used when plot_type = 'age'"""

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

def build_3d_model(chb, model_type, h_mud, h, w, bth, dcr, dx, delta_s, dt, starttime, endtime, diff_scale, v_fine, v_coarse, xmin, xmax, ymin, ymax):
    """function for building 3D model from set of centerlines (that are part of a ChannelBelt object)

    :param model_type: model type ('fluvial' or 'submarine')
    :param h_mud: maximum thickness of overbank deposit
    :param h: channel depth
    :param w: channel width
    :param bth: thickness of channel sand (only used in submarine models)
    :param dcr: critical channel depth where sand thickness goes to zero (only used in submarine models)
    :param dx: cell size in x and y directions
    :param delta_s: sampling distance alogn centerlines
    :param starttime: age of centerline that will be used as the first centerline in the model
    :param endtime: age of centerline that will be used as the last centerline in the model
    :param xmin: minimum x coordinate that defines the model domain; if xmin is set to zero, 
    a plot of the centerlines is generated and the model domain has to be defined by clicking its upper left and lower right corners
    :param xmax: maximum x coordinate that defines the model domain
    :param ymin: minimum y coordinate that defines the model domain
    :param ymax: maximum y coordinate that defines the model domain
    :param diff_scale: diffusion length scale (for overbank deposition)
    :param v_fine: deposition rate of fine sediment, in m/year (for overbank deposition)
    :param v_coarse: deposition rate of coarse sediment, in m/year (for overbank deposition)
    :return chb_3d: a ChannelBelt3D object
    :return xmin, xmax, ymin, ymax: x and y coordinates that define the model domain (so that they can be reused later)"""

    sclt = np.array(chb.cl_times)
    ind1 = np.where(sclt >= starttime)[0][0] 
    ind2 = np.where(sclt <= endtime)[0][-1]
    sclt = sclt[ind1:ind2+1]
    channels = chb.channels[ind1:ind2+1]
    cot = np.array(chb.cutoff_times)
    if (len(cot)>0) & (len(np.where(cot >= starttime)[0])>0) & (len(np.where(cot <= endtime)[0])>0):
        cfind1 = np.where(cot >= starttime)[0][0] 
        cfind2 = np.where(cot <= endtime)[0][-1]
        cot = cot[cfind1:cfind2+1]
        cutoffs = chb.cutoffs[cfind1:cfind2+1]
    else:
        cot = []
        cutoffs = []
    n_steps = len(sclt) # number of events
    if xmin == 0: # plot centerlines and define model domain
        plt.figure(figsize=(15,4))
        maxX, minY, maxY = 0, 0, 0
        for i in range(n_steps): # plot centerlines
            plt.plot(channels[i].x, channels[i].y, 'k')
            maxX = max(maxX, np.max(channels[i].x))
            maxY = max(maxY, np.max(channels[i].y))
            minY = min(minY, np.min(channels[i].y))
        plt.axis([0, maxX, minY-10*w, maxY+10*w])
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
    cutoff_levels = np.nan * np.zeros((n_steps, 1))
    # create initial topography:
    x1 = np.linspace(0, iwidth-1, iwidth)
    y1 = np.linspace(0, iheight-1, iheight)
    xv, yv = np.meshgrid(x1,y1)
    z1 = channels[0].z
    z1 = z1[(channels[0].x > xmin) & (channels[0].x < xmax)]
    topoinit = z1[0] - ((z1[0] - z1[-1]) / (xmax - xmin)) * xv * dx # initial (sloped) topography
    topo[:,:,0] = topoinit.copy()
    surf = topoinit.copy()
    facies[0] = np.NaN
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
        if i == 0:
            cl_dist_prev = cl_dist
        # erosion:
        surf = np.minimum(surf,erosion_surface(h,w/dx,cl_dist,z_map))
        topo[:,:,3*i] = surf # erosional surface
        dists[:,:,i] = cl_dist # distance map
        zmaps[:,:,i] = z_map # map of closest channel elevation
        facies[3*i] = np.NaN # array for facies code

        if model_type == 'fluvial':
            pb = point_bar_surface(cl_dist,z_map,h,w/dx)
            th = np.maximum(surf,pb)-surf
            th[cl_dist > 1.0 * w/dx] = 0 # eliminate sand outside of channel
            th[th<0] = 0 # eliminate negative thickness values
            surf = surf+th # update topographic surface with sand thickness
            topo[:,:,3*i+1] = surf # top of sand
            facies[3*i+1] = 1 # facies code for point bar sand
            E_max = z_map + h_mud[i]
            levee = fluvial_levee(cl_dist, surf, E_max, w/dx, diff_scale, v_fine, v_coarse, dt)
            surf = surf + levee # mud/levee deposition 
            topo[:,:,3*i+2] = surf # top of levee
            facies[3*i+2] = 2 # facies code for overbank
            channels3D.append(Channel(x1-xmin, y1-ymin, z1, w, h))

        if model_type == 'submarine':
            th, relief = sand_surface(surf, bth, dcr, z_map, h) # sandy channel deposit
            th[th < 0] = 0 # eliminate negative thickness values
            ws = w * (dcr/h)**0.5 # channel width at the top of the channel deposit
            th[cl_dist > 1.0 * ws/dx] = 0 # eliminate sand outside of channel
            surf = surf+th # update topographic surface with sand thickness
            topo[:,:,3*i+1] = surf # top of sand
            facies[3*i+1] = 1 # facies code for channel sand
            # need to blur z-map so that levees don't have artefacts:
            blurred = filters.gaussian(z_map, sigma=(50, 50), truncate=3.5, multichannel=False)
            E_max = blurred + h_mud[i]
            levee = submarine_levee(h_mud[i], cl_dist, surf, E_max, w/dx, diff_scale, v_fine, v_coarse, dt)
            surf = surf + levee # mud/levee deposition
            topo[:,:,3*i+2] = surf # top of levee
            facies[3*i+2] = 2 # facies code for overbank 
            channels3D.append(Channel(x1-xmin, y1-ymin, z1, w, h))

        cl_dist_prev = cl_dist.copy()
    topo = np.concatenate((np.reshape(topoinit,(iheight,iwidth,1)),topo),axis=2) # add initial topography to array
    strat = topostrat(topo) # create stratigraphic surfaces
    strat = np.delete(strat, np.arange(3*n_steps+1)[1::3], 2) # get rid of unnecessary stratigraphic surfaces (duplicates)
    facies = np.delete(facies, np.arange(3*n_steps)[::3]) # get rid of unnecessary facies layers (NaNs)
    if model_type == 'fluvial':
        facies_code = {1:'point bar', 2:'levee'}
    if model_type == 'submarine':
        facies_code = {1:'channel sand', 2:'levee'}
    chb_3d = ChannelBelt3D(model_type, topo, strat, facies, facies_code, dx, channels3D)
    # return chb_3d, xmin, xmax, ymin, ymax, dists, cutoff_dists_all, cutoff_levels, zmaps
    return chb_3d, xmin, xmax, ymin, ymax, dists, zmaps

def resample_centerline(x,y,z,deltas):
    '''resample centerline so that 'deltas' is roughly constant, using parametric 
    spline representation of curve; note that there is *no* smoothing

    :param x: x-coordinates of centerline
    :param y: y-coordinates of centerline
    :param z: z-coordinates of centerline
    :param deltas: distance between points on centerline
    :return x: x-coordinates of resampled centerline
    :return y: y-coordinates of resampled centerline
    :return z: z-coordinates of resampled centerline
    :return dx: dx of resampled centerline
    :return dy: dy of resampled centerline
    :return dz: dz of resampled centerline
    :return s: s-coordinates of resampled centerline'''

    dx, dy, dz, ds, s = compute_derivatives(x,y,z) # compute derivatives
    tck, u = scipy.interpolate.splprep([x,y,z],s=0) 
    unew = np.linspace(0,1,1+int(round(s[-1]/deltas))) # vector for resampling
    out = scipy.interpolate.splev(unew,tck) # resampling
    x, y, z = out[0], out[1], out[2] # assign new coordinate values
    dx, dy, dz, ds, s = compute_derivatives(x,y,z) # recompute derivatives
    return x,y,z,dx,dy,dz,ds,s

def migrate_one_step(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma):
    '''migrate centerline during one time step, using the migration computed as in Howard & Knutson (1984)

    :param x: x-coordinates of centerline
    :param y: y-coordinates of centerline
    :param z: z-coordinates of centerline
    :param W: channel width
    :param kl: migration rate (or erodibility) constant (m/s)
    :param dt: duration of time step (s)
    :param k: constant for calculating the exponent alpha (= 1.0)
    :param Cf: dimensionless Chezy friction factor
    :param D: channel depth
    :param omega: constant in Howard & Knutson equation (= -1.0)
    :param gamma: constant in Howard & Knutson equation (= 2.5)
    :return x: new x-coordinates of centerline after migration
    :return y: new y-coordinates of centerline after migration
    '''

    ns=len(x)
    curv = compute_curvature(x,y)
    dx, dy, dz, ds, s = compute_derivatives(x,y,z)
    sinuosity = s[-1]/(x[-1]-x[0])
    curv = W*curv # dimensionless curvature
    R0 = kl*curv # simple linear relationship between curvature and nominal migration rate
    alpha = k*2*Cf/D # exponent for convolution function G
    R1 = compute_migration_rate(pad,ns,ds,alpha,omega,gamma,R0)
    R1 = sinuosity**(-2/3.0)*R1
    # calculate new centerline coordinates:
    dy_ds = dy[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    dx_ds = dx[pad1:ns-pad+1]/ds[pad1:ns-pad+1]
    # adjust x and y coordinates (this *is* the migration):
    x[pad1:ns-pad+1] = x[pad1:ns-pad+1] + R1[pad1:ns-pad+1]*dy_ds*dt  
    y[pad1:ns-pad+1] = y[pad1:ns-pad+1] - R1[pad1:ns-pad+1]*dx_ds*dt 
    return x,y

def migrate_one_step_w_bias(x,y,z,W,kl,dt,k,Cf,D,pad,pad1,omega,gamma):
    ns=len(x)
    curv = compute_curvature(x,y)
    dx, dy, dz, ds, s = compute_derivatives(x,y,z)
    sinuosity = s[-1]/(x[-1]-x[0])
    curv = W*curv # dimensionless curvature
    R0 = kl*curv # simple linear relationship between curvature and nominal migration rate
    alpha = k*2*Cf/D # exponent for convolution function G
    R1 = compute_migration_rate(pad,ns,ds,alpha,omega,gamma,R0)
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
        si2 = np.hstack((np.array([0]),np.cumsum(ds[i-1::-1])))  # distance along centerline, backwards from current point 
        G = np.exp(-alpha*si2) # convolution vector
        R1[i] = omega*R0[i] + gamma*np.sum(R0[i::-1]*G)/np.sum(G) # main equation
    return R1

def compute_derivatives(x,y,z):
    """function for computing first derivatives of a curve (centerline)
    x,y are cartesian coodinates of the curve
    outputs:
    dx - first derivative of x coordinate
    dy - first derivative of y coordinate
    ds - distances between consecutive points along the curve
    s - cumulative distance along the curve"""
    dx = np.gradient(x) # first derivatives
    dy = np.gradient(y)   
    dz = np.gradient(z)   
    ds = np.sqrt(dx**2+dy**2+dz**2)
    s = np.hstack((0,np.cumsum(ds[1:])))
    return dx, dy, dz, ds, s

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
    ddx = np.gradient(dx) # second derivatives 
    ddy = np.gradient(dy) 
    curvature = (dx*ddy-dy*ddx)/((dx**2+dy**2)**1.5)
    return curvature

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
    """function for finding diagonal indices with k offset
    [from https://stackoverflow.com/questions/10925671/numpy-k-th-diagonal-indices]"""
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

def dist_map(x,y,z,xmin,xmax,ymin,ymax,dx,delta_s):
    """function for centerline rasterization and distance map calculation
    :param x: x coordinates of centerline
    :param y: y coordinates of centerline
    :param z: z coordinates of centerline
    :param xmin: minimum x coordinate that defines the area of interest
    :param xmax: maximum x coordinate that defines the area of interest
    :param ymin: minimum y coordinate that defines the area of interest
    :param ymax: maximum y coordinate that defines the area of interest
    :param dx: gridcell size (m)
    :param delta_s: distance between points along centerline (m)
    :return cl_dist: distance map (distance from centerline)
    :return x_pix: x pixel coordinates of the centerline
    :return y_pix: y pixel coordinates of the centerline
    :return z_pix: z pizel coordinates of the centerline
    :return s_pix: along-channel distance in pixels
    :return z_map: map of reference channel thalweg elevation (elevation of closest point along centerline)
    :return x: x centerline coordinates clipped to the 3D model domain
    :return y: y centerline coordinates clipped to the 3D model domain
    :return z: z centerline coordinates clipped to the 3D model domain"""
    y = y[(x>xmin) & (x<xmax)]
    z = z[(x>xmin) & (x<xmax)]
    x = x[(x>xmin) & (x<xmax)] 
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
    x_pix,y_pix = order_cl_pixels(x_pix,y_pix)
    # This next block of code is kind of a hack. Looking for, and eliminating, 'bad' pixels.
    img = np.array(img)
    img = img[:,:,0]
    img[img==255] = 1 
    img1 = morphology.binary_dilation(img, morphology.square(2)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix,y_pix) 
    img1 = morphology.binary_dilation(img, np.array([[1,0,1],[1,1,1]],dtype=np.uint8)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix,y_pix)
    img1 = morphology.binary_dilation(img, np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8)).astype(np.uint8)
    if len(np.where(img1==0)[0])>0:
        x_pix, y_pix = eliminate_bad_pixels(img,img1)
        x_pix,y_pix = order_cl_pixels(x_pix,y_pix)
    #redo the distance calculation (because x_pix and y_pix do not always contain all the points in cl):
    cl[cl==0] = 1
    cl[y_pix,x_pix] = 0
    cl_dist, inds = ndimage.distance_transform_edt(cl, return_indices=True)
    dx,dy,dz,ds,s = compute_derivatives(x,y,z)
    dx_pix = np.diff(x_pix)
    dy_pix = np.diff(y_pix)
    ds_pix = np.sqrt(dx_pix**2+dy_pix**2)
    s_pix = np.hstack((0,np.cumsum(ds_pix)))
    f = scipy.interpolate.interp1d(s,z)
    snew = s_pix*s[-1]/s_pix[-1]
    if snew[-1]>s[-1]:
        snew[-1]=s[-1]
    snew[snew<s[0]]=s[0]
    z_pix = f(snew)
    # create z_map:
    z_map = np.zeros(np.shape(cl_dist)) 
    z_map[y_pix,x_pix]=z_pix
    xinds=inds[1,:,:]
    yinds=inds[0,:,:]
    for i in range(0,len(x_pix)):
        z_map[(xinds==x_pix[i]) & (yinds==y_pix[i])] = z_pix[i]
    return cl_dist, x_pix, y_pix, z_pix, s_pix, z_map, x, y, z

def erosion_surface(h,w,cl_dist,z):
    """function for creating a parabolic erosional surface
    :param h: geomorphic channel depth (m)
    :param w: geomorphic channel width (in pixels, as cl_dist is also given in pixels)
    :param cl_dist: distance map (distance from centerline)
    :param z: reference elevation (m)
    :return surf: map of the erosional surface (m)
    """
    surf = z + (4*h/w**2)*(cl_dist+w*0.5)*(cl_dist-w*0.5)
    return surf

def point_bar_surface(cl_dist,z,h,w):
    """function for creating a Gaussian-based point bar surface
    used in 3D fluvial model
    :param cl_dist: distance map (distance from centerline)
    :param z: reference elevation (m)
    :param h: channel depth (m)
    :param w: channel width, in pixels, as cl_dist is also given in pixels
    :return pb: map of the Gaussian surface that can be used to from a point bar deposit (m)"""
    pb = z-h*np.exp(-(cl_dist**2)/(2*(w*0.33)**2))
    return pb

def sand_surface(surf, bth, dcr, z_map, h):
    """function for creating the top horizontal surface sand-rich deposit in the bottom of the channel
    used in 3D submarine channel models
    :param surf: current geomorphic surface
    :param bth: thickness of sand deposit in axis of channel (m)
    :param dcr: critical channel depth, above which there is no sand deposition (m)
    :param z_map: map of reference channel thalweg elevation (elevation of closest point along centerline)
    :param h: channel depth (m)
    :return th: thickness map of sand deposit (m)
    :return relief: map of channel relief (m)"""
    relief = np.abs(surf - z_map + h)
    relief = np.abs(relief - np.amin(relief))
    th = bth * (1 - relief/dcr) # bed thickness inversely related to relief
    th[th<0] = 0.0 # set negative th values to zero
    return th, relief

def fluvial_levee(cl_dist, topo, E_max, w, diff_scale, v_fine, v_coarse, dt):
    """function for creating a levee layer in a fluvial 3D model
    based on the diffusion-based overbank deposition model of Howard, 1992
    :param h_mud: maximum thickness of overbank deposit (specific to time step)
    :param cl_dist: distance map (distance from centerline)
    :param topo: current topographic surface
    :param E_max: maximum thickness of overbank deposit (specific to location)
    :param w: channel width
    :param diff_scale: diffusion length scale
    :param v_fine: deposition rate of fine sediment, in m/year (for overbank deposition)
    :param v_coarse: deposition rate of coarse sediment, in m/year (for overbank deposition)
    :param dt: time step (in seconds)
    :return levee: thickness of levee layer (same size as 'topo')"""
    dep_rate = (E_max - topo) * (v_fine + v_coarse * np.exp(-cl_dist/diff_scale))
    dep_rate[cl_dist < 0.5*w] = 0  # get rid of the mud in the active channel
    dep_rate[dep_rate < 0] = 0
    levee = dep_rate * (dt/(365*24*60*60))
    return levee

def submarine_levee(h_mud, cl_dist, topo, E_max, w, diff_scale, v_fine, v_coarse, dt):
    """function for creating a levee layer in a submarine 3D model
    based on the diffusion-based overbank deposition model of Howard, 1992
    :param h_mud: maximum thickness of overbank deposit (specific to time step)
    :param cl_dist: distance map (distance from centerline)
    :param topo: current topographic surface
    :param E_max: maximum thickness of overbank deposit (specific to location)
    :param w: channel width
    :param diff_scale: diffusion length scale
    :param v_fine: deposition rate of fine sediment, in m/year (for overbank deposition)
    :param v_coarse: deposition rate of coarse sediment, in m/year (for overbank deposition)
    :param dt: time step (in seconds)
    :return levee: thickness of levee layer (same size as 'topo')"""
    dep_rate = (E_max - topo) * (v_fine + v_coarse * np.exp(-cl_dist/diff_scale))
    dep_rate[dep_rate < 0] = 0
    levee = dep_rate * (dt/(365*24*60*60))
    surf3 = h_mud + (4*h_mud / w**2) * (cl_dist + w*0.5) * (cl_dist - w*0.5) # 'erosional' surface
    levee = np.minimum(levee, surf3) # get rid of the mud in the axis of the active channel
    return levee

def topostrat(topo):
    """function for converting a stack of geomorphic surfaces into stratigraphic surfaces
    :param topo: 3D numpy array of geomorphic surfaces, with the oldest at index 0
    :return strat: 3D numpy array of stratigraphic surfaces, with the oldest at index 0
    assumption is that topographic array has oldest surface at '0' z (third) index and therefore needs to be flipped twice
    """
    strat = np.minimum.accumulate(topo[:, :, ::-1], axis=2)[:, :, ::-1] # this eliminates the 'for' loop and is therefore faster
    return strat

def eliminate_bad_pixels(img, img1):
    """function for removing 'bad' pixels along channel centerline
    :param img: black-and-white image of channel centerline (centerline pixels are 0)
    :param img1: dilated version of centerline image 'img'
    :return x_pix: cleaned array of x pixel xoordinates
    :return y_pix: cleaned array of y pixel xoordinates"""
    x_ind = np.where(img1==0)[1][0]
    y_ind = np.where(img1==0)[0][0]
    img[y_ind:y_ind+2,x_ind:x_ind+2] = np.ones(1,).astype(np.uint8)
    all_labels = measure.label(img,background=1,connectivity=2)
    cl=all_labels.copy()
    cl[cl==2]=0
    cl[cl>0]=1
    y_pix,x_pix = np.where(cl==1)
    return x_pix, y_pix

def order_cl_pixels(x_pix,y_pix):
    '''function for ordering pixels along a channel centerline, starting on the left side
    :param x_pix: unordered x pixel coordinates of the centerline
    :param y_pix: unordered y pixel coordinates of the centerline
    :return x_pix: ordered x pixel coordinates of the centerline
    :return y_pix: ordered y pixel coordinates of the centerline'''
    dist = distance.cdist(np.array([x_pix,y_pix]).T,np.array([x_pix,y_pix]).T)
    dist[np.diag_indices_from(dist)]=100.0
    ind = np.argmin(x_pix) # select starting point on left side of image
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