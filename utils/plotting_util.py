import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pylab import plt

from utils import optimization_constraints

import os
import sys
from utils import parameter_util
import mxnet as mx


def plot_with_uncertainty(x, y, y11=None, y12=None, y21=None, y22=None, color='r', linestyle='-', label='', alpha1=1.0, alpha2=0.25, pl=plt, plot_percentiles=False):
    x=x.flatten()
    y=y.flatten()
    pl.plot(x, y, color=color, linestyle=linestyle, label=label,alpha=alpha1)
    if not y11 is None and plot_percentiles:
        y11=y11.flatten()
        y12=y12.flatten()
        pl.fill_between(x.flatten(), y11, y12, color=color, alpha=alpha2, linestyle=linestyle)
    if not y21 is None and plot_percentiles:
        y21=y21.flatten()
        y22=y22.flatten()
        pl.fill_between(x.flatten(), y21, y22, color=color, alpha=alpha2, linestyle=linestyle)



def plot_latent_space_direct(m, h_mean, y_latent, x_ref=None, cost_function=None, savename=None, samples_l=None, inducing=False, highlight=None, config={}):
    dim_h = h_mean.shape[1]
    bounds = np.array([[min(h_mean[:,i]), max(h_mean[:,i])] for i in range(dim_h)])
    #print(bounds)
    _min, _max = np.amin(y_latent), np.amax(y_latent)
    Nt=100
    v1 = np.linspace(bounds[0,0],bounds[0,1], Nt)
    v2 = np.linspace(bounds[1,0],bounds[1,1], Nt)
    xv0, xv1 = np.meshgrid(v1, v2, sparse=False, indexing='ij')
    extra_dims = 0
    if dim_h>2:
        extra_dims = dim_h - 2
        
    xt = np.array([xv0.reshape((-1)), xv1.reshape((-1))] + [np.zeros(xv0.shape).reshape((-1)) for i in range(extra_dims)]).T
    yt, yvt, xpredt = m.predict(mx.nd.array(xt, dtype=np.float64), diag=True)[:3]
    #import pdb; pdb.set_trace()
    #yvt = np.diag(yvt)
    #_min, _max = np.amin(np.r_[y_latent, yt]), np.amax(np.r_[y_latent, yt])
    _min, _max = np.amin(y_latent), np.amax(y_latent)
    f, (ax) = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15.3,3.5))

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(h_mean[:,0], h_mean[:,1])
    
    xy = np.dstack((triang.x[triang.triangles], triang.y[triang.triangles]))  # shape (ntri,3,2)
    twice_area = np.cross(xy[:,1,:] - xy[:,0,:], xy[:,2,:] - xy[:,0,:])  # shape (ntri)
    mask = twice_area < 1e-10  # shape (ntri)

    if np.any(mask):
        triang.set_mask(mask)
    interpolator = tri.LinearTriInterpolator(triang, y_latent.reshape(-1))
    yt_latent = interpolator(xv0, xv1)

    cs = ax[0].contourf(xv0, xv1, yt_latent.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = _min, vmax = _max, extend='both')
    ax[0].set_title("Function value projection")
    cs.cmap.set_over('white')
    cs.cmap.set_under('black')
    cs.changed()
    plt.subplots_adjust(bottom=0.10)

    cbar_ax = f.add_axes([0.2, 0.02, 0.35, 0.05])
    cb = plt.colorbar(cs, orientation="horizontal",cax=cbar_ax) 


    if cost_function is not None:
        xpredt = m.predict(mx.nd.array(xt, dtype=np.float64))[2]
        ypredt = cost_function(xpredt)
        ypredt[ypredt > _max] = _max
        ax[1].contourf(xv0, xv1, ypredt.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = _min, vmax = _max, extend='both')
        ax[1].set_title("Function value for decoded point from latent space")

    
    
    yt = np.clip(yt, _min, _max)
    
    cs = ax[2].contourf(xv0, xv1, yt.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10))#, vmin = _min, vmax = _max, extend='both')
    ax[2].set_title("GPLVM fit on latent space")
    cs.cmap.set_over('white')
    cs.cmap.set_under('black')
    cs.changed()
    
    if samples_l is not None:
        ax[2].scatter(samples_l[:,0], samples_l[:,1], marker='+', color='r', alpha=0.1)
        #import pdb; pdb.set_trace()
        if inducing:
            try:
                samples_i = m.likelihood.params.get('inducing_inputs').data().asnumpy()
                ax[2].scatter(samples_i[:,0], samples_i[:,1], marker='+', color='b', alpha=0.05)
            except:
                pass
        if highlight is not None:
            ax[2].scatter(highlight[:,0], highlight[:,1], marker='o', color='r')
            ax[0].scatter(highlight[:,0], highlight[:,1], marker='o', color='r')
            ax[4].scatter(highlight[:,0], highlight[:,1], marker='o', color='r')
            if m.X_try is not None:
                ax[2].scatter(m.X_try[:,0], m.X_try[:,1], marker='x', color='y')
                ax[0].scatter(m.X_try[:,0], m.X_try[:,1], marker='x', color='y')
                ax[4].scatter(m.X_try[:,0], m.X_try[:,1], marker='x', color='y')
        #c = WithinConvexSet(h_mean)
        #plot_linears(ax[0], h_mean, c.A, c.b, bounds)
        
    if x_ref is not None:
        #print(x_ref.shape)
        z_ref = m.predict(mx.nd.array(x_ref, dtype=np.float64))[3]
        ax[0].scatter(z_ref[0,0], z_ref[0,1], marker='+', color='b')
        #ax[1].scatter(z_ref[0,0], z_ref[0,1], marker='+', color='r')
     
        
    
    if True:
        cs = ax[3].contourf(xv0, xv1, np.sqrt(yvt).reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = 0)
        ax[3].set_title("GPLVM std on latent space")
        cs.changed()
        
        if samples_l is not None:
            ax[3].scatter(samples_l[:,0], samples_l[:,1], marker='+', color='r')
            samples_i = m.likelihood.params.get('inducing_inputs').data().asnumpy()
            ax[3].scatter(samples_i[:,0], samples_i[:,1], marker='x', color='b')    



    if False:
        optimized_space =  config.get('OPTIMIZED_SPACE', parameter_util.OPTIMIZED_SPACE)
        if optimized_space == parameter_util.CONVEX:
            asd = optimization_constraints.WithinConvexSet(h_mean, batch_size=1)
        elif optimized_space == parameter_util.STABLE:
            asd = optimization_constraints.WithinStableSet(h_mean,  m, batch_size=1)
        else:
            asd = optimization_constraints.WithinData(h_mean,batch_size=1) # 
        
        Nt = 150
        v1 = np.linspace(bounds[0,0],bounds[0,1], Nt)
        v2 = np.linspace(bounds[1,0],bounds[1,1], Nt)
        xv0, xv1 = np.meshgrid(v1, v2, sparse=False, indexing='ij')

        xt = np.array([xv0.reshape((-1)), xv1.reshape((-1))]).T
        vals = []
        for i in range(xt.shape[0]):
            vals += [np.all(asd.f(xt[i:i+1,:]) < asd.bounds[1])]
        
        vals=np.array(vals)
        
        #vals[vals < asd.bounds[1]] = 0
        
        #vals[vals > asd.bounds[1]] = 1
        
        cs = ax[4].contourf(xv0, xv1, vals.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = 0)
        ax[4].set_title("optimizedspace")
        if highlight is not None:
            ax[4].scatter(highlight[:,0], highlight[:,1], marker='o', color='r')
            if m.X_try is not None:
                ax[2].scatter(m.X_try[:,0], m.X_try[:,1], marker='x', color='y')


    cbar_ax = f.add_axes([0.75, 0.02, 0.25, 0.05])
    cb = plt.colorbar(cs, orientation="horizontal",cax=cbar_ax) 

    #plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0.15, right = 1, left = 0, hspace = 0.0, wspace = 0.0)
    
    plt.margins(0,0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(savename, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()
    #import pdb; pdb.set_trace()
    #plt.show()

def plot_linears(ax, data, A, b, bounds):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    for i in range(A.shape[0]):
        f_t = lambda x: (b[i] - A[i,0]*x)/A[i,1]
        ax.plot([mins[0], maxs[0] ], [f_t(mins[0]), f_t(maxs[0])])
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])

def plot_model_base_2d(m, bounds, h_mean, y_latent, samples1=None, samples2=None, highlight_last=False, x_ref=None, cost_function=None, plot_weights=True):
    _min, _max = np.amin(y_latent), np.amax(y_latent)
    Nt=100
    v1 = np.linspace(bounds[0,0],bounds[0,1], Nt)
    v2 = np.linspace(bounds[1,0],bounds[1,1], Nt)
    xv0, xv1 = np.meshgrid(v1, v2, sparse=False, indexing='ij')

    xt = np.array([xv0.reshape((-1)), xv1.reshape((-1))]).T
    yt, yvt, xpredt = m.predict(mx.nd.array(xt, dtype=np.float64))[:3]

    f, (ax) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(13.3,3))

    ax[0].contourf(xv0, xv1, yt.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = _min, vmax = _max)
    ax[0].set_title("GPLVM fit on latent space")

    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(h_mean[:,0], h_mean[:,1])
    interpolator = tri.LinearTriInterpolator(triang, y_latent)
    yt_latent = interpolator(xv0, xv1)

    ax[1].contourf(xv0, xv1, yt_latent.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = _min, vmax = _max)
    ax[1].set_title("Function value projection")

    if samples1 is not None:
        ax[0].scatter(samples1[:,0], samples1[:,1], marker='+', color='y')
        #ax[1].scatter(samples1[:,0], samples1[:,1], marker='+', color='y')
        if highlight_last:
            ax[0].scatter(samples1[-1,0], samples1[-1,1], marker='o', color='b')
            #ax[1].scatter(samples1[-1,0], samples1[-1,1], marker='o', color='b')
    if samples2 is not None:
        ax[0].scatter(samples2[:,0], samples2[:,1], marker='+', color='b')
        #ax[1].scatter(samples2[:,0], samples2[:,1], marker='+', color='b')
    if x_ref is not None:
        #print(x_ref.shape)
        z_ref = m.predict(mx.nd.array(x_ref.reshape((1,-1)), dtype=np.float64))[3]
        ax[0].scatter(z_ref[0,0], z_ref[0,1], marker='+', color='r')
        #ax[1].scatter(z_ref[0,0], z_ref[0,1], marker='+', color='r')

    if cost_function is not None:
        _, _, xpredt = m.predict(mx.nd.array(xt, dtype=np.float64))[:3]
        ypredt = cost_function(xpredt)
        ax[2].contourf(xv0, xv1, ypredt.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = _min, vmax = _max)
        ax[2].set_title("Function value for decoded point from latent space")
    if plot_weights:
        weights = m.compute_weights(x=mx.nd.array(xt, dtype=np.float64)).asnumpy()
        print("Weight statistics: min {}, max {}".format(min(weights), max(weights)))
        ax[3].contourf(xv0, xv1, weights.reshape((Nt,Nt)),cmap=plt.cm.get_cmap('binary_r', 10), vmin = _min, vmax = _max)
        ax[3].set_title("Weights in the latent space")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_latent_space(m, H_mean, y_latent, samples1=None, samples2=None, highlight_last=False, x_ref=None, cost_function=None, plot_weights=False):
    #Compute the bounds
    dim_h = H_mean.shape[1]
    bounds = np.array([[min(H_mean[:,i]), max(H_mean[:,i])] for i in range(dim_h)])
    if dim_h == 1:
        pass
    elif dim_h == 2:
        plot_model_base_2d(m, bounds, H_mean, y_latent, samples1=samples1, samples2=samples2, highlight_last=highlight_last,  x_ref=x_ref, cost_function=cost_function, plot_weights=plot_weights)


def plot_encoded_in_grid(m, latent_bounds, x_ref=None, Ni=35, savename=None):
    #Show how training samples populate the latent value space:
    fig, axarr = plt.subplots(Ni, Ni, figsize=(20,20))
    for i in range(axarr.shape[0]):
        for j in range(axarr.shape[1]):
            axarr[i,j].get_xaxis().set_ticks([])
            axarr[i,j].get_yaxis().set_ticks([])
    plt.subplots_adjust(hspace=0, wspace=0)

    x_grid = latent_bounds[0,0] + (np.arange(Ni)+0.5)*(latent_bounds[0,1] - latent_bounds[0,0])/Ni
    y_grid = latent_bounds[1,0] + (np.arange(Ni)+0.5)*(latent_bounds[1,1] - latent_bounds[1,0])/Ni
    xv0, xv1 = np.meshgrid(x_grid, y_grid, sparse=False, indexing='ij')

    xt = np.array([xv0.reshape((-1)), xv1.reshape((-1))]).T
    xpredt = m.predict(mx.nd.array(xt, dtype=np.float64), diag=True)[2]
    x_ref_latent = m.predict(mx.nd.array(x_ref, dtype=np.float64), diag=True)[3]

    delta = np.array([latent_bounds[0,1] - latent_bounds[0,0], latent_bounds[1,1] - latent_bounds[1,0]])*0.5/Ni
    #import pdb; pdb.set_trace()
    if xpredt.shape[1] == 1:
        xpredt = xpredt[:,0,:,:]
    else:
        xpredt = np.rollaxis(xpredt, 1, 4)
    
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            axarr[Ni-j-1,i].imshow(xpredt[i*Ni+j,:], cmap='Greys')
            if(abs(x_ref_latent[:,0]-x) < delta[0] and abs(x_ref_latent[:,1]-y) < delta[1]):
                axarr[Ni-j-1,i].imshow(xpredt[i*Ni+j,:], cmap='Reds')
    if savename is None:
        plt.show()
    else:
        plt.savefig(savename, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()
