import numpy as np
import time


def logistic(a):
    """
    returns the logistic sigmoid \pi(a)
    Keyword arguments:
    a -- scalar or numpy array
    """
    expa = np.exp(a)
    res = expa / (1.0 + expa)
    if hasattr(a, "__iter__"):
        res[a>709.7] = 1.0 # np.exp will overflow and return inf for values larger 709.7.
    elif a>709.7:
        res = 1.0
    return res

def logit(a):
    return np.log(a)-np.log(1.0-a)

def logit_scalar_brent(a):
    if a!=0.5:
        # determine corresponding x_root
        from scipy.optimize import brentq
        def f(x):
            return a-logistic(x)    # we want to find a zero/root of f(x) using brent's method.
        return brentq(f, -10, 10, args=(), xtol=2e-12, rtol=8.8817841970012523e-16, maxiter=100, full_output=False, disp=True)

    else:
        return 0.0
    
def logit_brent(a):
    if hasattr(a, "__iter__"):
        f = np.vectorize(logit_scalar_brent)  # or use a different name if you want to keep the original f
        return f(a)
    else:
        return logit_scalar_brent(a)

class LogisticRegression(object):
    """
    Implements logistic regression classifier with the objective
    \sum_{n\in class_{1}} \log(\pi(x_n.dot(w))) + \sum_{n\in class_{0}} \log(1-\pi(x_n.dot(w))) + lambd/2 * w.T.dot(w)
    """

    def __init__(self, lambd=1e-3, tol=1e-5, max_iter=100, learning_rate=1e-4, decay_rate=1e-5, optimizer="IRLS", verbose=False, debug=False):
        """
        Keyword arguments:
        lambd     -- regularization paramter for L2 norm of w (scalar or numpy 1D array with length equal to the number of dimensions) (default: 1e-5)
        tol       -- tolerance of the optimizer (default: 1e-5)
        max_iter  -- maximum number of interations of the optimizer (default: 100)
        optmizer  -- "IRLS" for Newton Raphson/IRLS or "steep" for steepest descent (default: "IRLS")
        verbose   -- Boolean indicator (default: False)
        """
        self.w = None # create a placeholer for the weights w
        self.class_labels = None # crete a placeholder for the list of class labels
        self.lambd = lambd
        self.tol = tol
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.decay_rate = decay_rate
        self.debug = debug
        
    def fit(self, X, y, w_init=None):
        """
        minimize the objective
        \sum_{n\in class_{1}} \log(\pi(x_n.dot(w))) + \sum_{n\in class_{0}} \log(1-\pi(x_n.dot(w))) + lambd/2 * w.T.dot(w)
        """
            
        self.class_labels = np.unique(y)
        if len(self.class_labels)>2:
            raise Exception("too many classes. This logistic regression class only implements binary classification.")
        if w_init is None:
            self.w = np.zeros((X.shape[1], 1)) # zero-init w
        else:
            self.w = w_init
        num_iter = 0
        objective_last = np.inf# initialize the objective to a large number
        gradient_last = np.inf # initialize the gradient to a large number
        # Newton-Raphson / IRLS updates
        if self.verbose or self.debug:
            t0 = time.time()
            w_ret = [self.w]
            gradient_ret = []
            objective_ret = [self.objective(y,X)]

        while (objective_last>0.0) and (np.sqrt(gradient_last*gradient_last).sum() > self.tol) and (num_iter<self.max_iter):
            gradient_last = self.perform_update(X=X,y=y)
            if self.verbose or self.debug:
                objective = self.objective(y,X) # compute the objective
                objective_ret.append(objective)
                w_ret.append(self.w)
                gradient_ret.append(gradient_last)
                # if (self.optimizer=="IRLS") & ((objective_last - objective) < -self.tol) :
                #     print("objective is getting significantly worse")
                objective_last = objective
                time_sec = time.time() - t0
                if self.verbose and (np.mod(num_iter,1000) == 0): 
                    print( "[iteration {:}, {:.4f}s]: objective: {:.3e}, gradient l2 norm : {:.3e}".format(num_iter, time_sec, objective, np.sqrt(gradient_last*gradient_last).sum()))
            num_iter += 1
        if self.verbose or self.debug:
            gradient_ret.append(self.gradient(X=X,y=y))
            print( "[iteration {:}, {:.4f}s]: objective: {:.3e}, gradient l2 norm : {:.3e}".format(num_iter, time_sec, objective, np.sqrt(gradient_last*gradient_last).sum()))
            return self, {"objective": objective_ret, "w":w_ret, "gradient": gradient_ret}
        return self
    
    def perform_update(self, X, y):
        """
        compute and perform a single step in the iterative optimization scheme
        """
        if self.optimizer == "IRLS":
            # compute the Iteratively Reweighted Least Squares update (equiv. Newton-Raphson)
            hessian = self.hessian(X=X, y=y)
            gradient = self.gradient(X=X, y=y)
            update = - np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        elif self.optimizer == "steep":
            # compute the steepest descent update
            gradient = self.gradient(X=X, y=y)
            update = -self.learning_rate * gradient
            self.learning_rate = max(1e-6,(1-self.decay_rate) * self.learning_rate)
        # Todo: maybe implement a backtracking line search.
        self.w = self.w + update
        return gradient
    
    def predict_proba(self, X, min_val=1e-15):
        """
        compute the probabilities of the lexicographically larger class label
        """
        z = logistic(np.dot(X, self.w))
        z[z<min_val] = min_val
        z[z>1-min_val] = 1-min_val
        return z
    
    def predict(self, X, threshold=0.5):
        """
        predict a class label using \pi(x)>=threshold
        """
        prediction =  np.array([self.class_labels[0]] * X.shape[0])[:,np.newaxis]
        prediction[self.predict_proba(X) >= threshold] = self.class_labels[1] 
        return prediction
    
    def objective(self, y, X):
        """
        L = \sum_{n\in class_{1}} \log(\pi(x_n.dot(w))) + \sum_{n\in class_{0}} \log(1-\pi(x_n.dot(w))) + lambd/2 * w.T.dot(w)
        """
        pi = self.predict_proba(X)
        log_0_pi = np.log(pi[y==self.class_labels[1]])
        log_1_pi = np.log(1.0-pi[y==self.class_labels[0]])
        loss = -log_0_pi.sum() - log_1_pi.sum() # this version is more stable for perfect prediction
        regularizer =  0.5 *  (self.lambd * self.w * self.w).sum()
        return loss  + regularizer
    
    def gradient(self, X, y):
        """
        compute the [D x 1] gradient vector
        \nabla w := [dL / dw_j for each j \in 1..D]
        """
        pi = self.predict_proba(X)
        return np.dot(X.T, pi-(y==self.class_labels[1]))+ self.lambd * self.w
    
    def hessian(self, X, y):
        """
        compute the [D x D] Hessian matrix
        \nabla^2 w := [d^2 L / (dw_i dw_j) for each i,j \in 1..D]
        """
        pi = self.predict_proba(X)
        return  (X * (pi * (1.0-pi))).T.dot(X) + self.lambd * np.eye(X.shape[1])