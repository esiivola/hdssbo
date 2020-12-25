from typing import Tuple, Dict, Callable, Optional
import numpy as np
import mxnet as mx

class WithinStableSet():
    def __init__(self, data, model, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.d = data.shape[1]
        self.model = model
        dists = self._compute_distance_mx(mx.nd.array(data, dtype=np.float64)).asnumpy().reshape(-1)
        max_dist = np.percentile(dists, 95)
        self.bounds=[np.zeros(self.batch_size), np.ones(self.batch_size)*max_dist]
    
    def _compute_distance_mx(self, X_mx):
        X_new = self.model._comp_logL.predict_mx(self.model._comp_logL.predict_mx(X_mx, diag=True, push=False)[2], diag=True, push=False)[3]
        return mx.nd.sqrt(mx.nd.sum(mx.nd.square(X_mx-X_new), axis=1))
   
    def f(self, x):
        y = []
        x = x.reshape((self.batch_size, self.d))
        for i in range(self.batch_size):
            y += [self._compute_distance_mx(mx.nd.array(x[i:i+1,:]  ,dtype=np.float64)).asnumpy()]
        ret = np.array(y).reshape((-1))
        return ret
    
    def jacobian(self, x):
        x = x.reshape((self.batch_size, self.d))
        dg = []
        #import pdb; pdb.set_trace()
        for i in range(self.batch_size):
            X_mx = mx.nd.array(x[i:i+1,:], dtype = np.float64)
            X_mx.attach_grad()

            with mx.autograd.record():
                distance = self._compute_distance_mx(X_mx)
                distance_g = mx.autograd.grad(distance, X_mx, retain_graph=True)
            dg += [distance_g[0].asnumpy().reshape(1,-1)]
        grad = block_diag(*[dg[i] for i in range(self.batch_size)])
        #import pdb; pdb.set_trace()
        return grad

class WithinConvexSet():
    def __init__(self, data, batch_size = 1):
        self.data = data
        self.batch_size = batch_size
        self.d = data.shape[1]
        convex_hull = ConvexHull(self.data)
        
        simplices = convex_hull.simplices
        
        center = np.mean(self.data, axis=0)[None,:]
        
        # Solve the arrays 
        A = np.empty((0,self.data.shape[1]))
        B = np.empty((0,1))
        #import pdb; pdb.set_trace()
        for simplex in simplices:
            X = self.data[simplex,:]
            b = np.ones((X.shape[0],1))
            a = la.inv(X) @ np.ones((X.shape[0],1))
            # choose sign:
            if np.dot(a, center)[0,0] > 1:
                a = -1*a
                b = -1*b
            A = np.r_[A, a.T]
            B = np.r_[B, b[:1]]
            b = B.reshape(-1)

        self.A_ = A.T

        self.A = A.T 
        
        self.jacob = block_diag(*[A.T for i in range(self.batch_size)]).T
            
        self.b = np.tile(b.reshape((1,-1)), (self.batch_size,1))
            
        self.bounds = [-np.Inf*np.ones(self.A.shape[1]*self.batch_size), np.zeros(self.A.shape[1]*self.batch_size)]

    def f(self, x):
        x = x.reshape((self.batch_size, self.d))
        return (x @ self.A - self.b).reshape(-1) # batch_size * num_planes 
    
    def jacobian(self, x):
        return self.jacob 
 

class WithinEllipsoid():
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.d = data.shape[1]
        self.bounds=[np.zeros(self.batch_size), np.ones(self.batch_size)]
        self._getMinVolEllipse()

    def _getMinVolEllipse(self, tolerance=0.01):
        """ Find the minimum volume ellipsoid which holds all the points
        
        Based on work by Nima Moshtagh
        http://www.mathworks.com/matlabcentral/fileexchange/9542
        and also by looking at:
        http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
        Which is based on the first reference anyway!
        
        Here, P is a numpy array of N dimensional points like this:
        P = [[x,y,z,...], <-- one point per line
             [x,y,z,...],
             [x,y,z,...]]
        
        Returns:
        (center, radii, rotation)
        
        """
        P = self.data
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(la.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = la.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = la.inv(np.dot(P.T, np.dot(np.diag(u), P)) - 
                   np.array([[a * b for b in center] for a in center])
                  ) / d
        self.center = center.reshape((self.d,1)) # np.tile(center, self.batch_size)
        self.E = A # block_diag(*[A for i in range(self.batch_size)])
    
    def f(self, x):
        x = x.reshape((self.batch_size, self.d)).T
        ret = np.diag((x-self.center).T @ self.E @ (x-self.center))
        return ret.reshape(-1) #self.batch_size,             size of E: latent dim times latent dim
    
    def jacobian(self, x): #batch times x.shape 
        x = x.reshape((self.batch_size, self.d)).T
        f = [2.0*self.E @ (x[:,i:i+1]-self.center) for i in range(self.batch_size) ] # each is of size d times 1
        return block_diag(*f).T