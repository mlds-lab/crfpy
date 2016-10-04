import numpy as np
from pystruct.learners import NSlackSSVM
import itertools as it
from scipy.misc import logsumexp
from scipy.optimize import minimize
from pyhmc import hmc
from rutils import *

class CRF:
    def __init__(self,lambda_0=0.0,n_parameters=1,max_iter=100,verbose=0,test_grad=False,objective="mle",method="bfgs",batch_size=100,lr=1.0,tol=1e-4):
        self.__dict__.update(locals())
    
    def accuracy(self,X,Y):
        accuracy = 0.0
        total = 0.0
        for x,y in zip(X,Y):
            y_hat = self.predict(X)
            accuracy += np.sum(y==y_hat)
            total += y.shape[0]
            
        return accuracy/total
        
    def fit(self,X,Y):
        if self.objective=="mle":
            self.fit_mle(X,Y)
        elif self.objective == "ssvm":
            self.fit_ssvm(X,Y)
        elif self.objective == "alpha_ssvm":
            self.fit_alpha_ssvm(X,Y)
        else:
            raise(ValueError("%s not implemented as a fit method."%self.method))
        
    def sufficient_statistics(self, x, y):
        raise NotImplementedError( "sufficient_statistics not implemented." )
        
    def expected_sufficient_statistics(self, return_logZ=False):
        raise NotImplementedError( "expected_sufficient_statistics not implemented." )
        
    def get_weight_vector(self):
        raise NotImplementedError( "get_weight_vector not implemented." )
        
    def set_weights(self, w):
        raise NotImplementedError( "set_weights not implemented." )
        
    def map_inference(self, X):
        raise NotImplementedError("map_inference not implemented.")
        
    def set_loss_augmented_weights(self, x, y, w):
        raise NotImplementedError("set_loss_augmented_weights not implemented.")
        
    def loss(self,y,y_hat):
        raise NotImplementedError("loss not implemented.")
        
    def deaugment(self,w):
        raise NotImplementedError("deaugment not implemented.")
        
    def devectorize_labels(self,y):
        return y
        
    def vectorize_labels(self,y):
        return y
        
    def joint_feature(self,x,y):
        y = self.devectorize_label(y)
        return self.sufficient_statistics(x,y)
        
    def inference(self,x,w):
        self.set_weights(w)
        return self.vectorize_label(self.map_inference([x])[0])
        
    def loss_augmented_inference(self,x,y,w,relaxed=False):
        y = self.devectorize_label(y)
        self.inference_calls += 1
        x_aug = self.set_loss_augmented_weights(x,y,w)
        y_hat,score = self.map_inference([x_aug],return_score=True)[0]
        # assert np.dot(self.joint_feature(x,y_hat),w) + self.loss(y,y_hat) == score
        if not np.isclose(np.dot(self.sufficient_statistics(x,y_hat),w) + self.loss(y,y_hat,vectorized_labels=False), score):
            print "Inference Error: scores don't match"
            w_aug = self.get_weight_vector()
            jf_aug = self.joint_feature(x_aug,y_hat)
            keyboard()
        return self.vectorize_label(y_hat)
        
    def log_likelihood(self,X,Y,SS=None,return_gradient=False,alpha=1.0,loss_augmented=False):
        # get parameter vector
        w = self.get_weight_vector()
        
        # If sufficient statistics not passed calculate them
        if SS is None:
            SS = []
            for x,y in zip(X,Y):
                SS.append(self.sufficient_statistics(x,y))

        # Initialize log_likelihood and gradients
        f = 0.0
        if return_gradient:
            avg_sufficient_statistics = np.zeros(self.n_parameters)
            
        # for each datacase calculate the expected sufficient statistics 
        # and increment the log_likelihood and gradients
        for s in range(len(SS)):
            sufficient_statistics = SS[s]
            if loss_augmented:
                x = self.set_loss_augmented_weights(X[s],Y[s],w,alpha=alpha)
            else:
                x = X[s]
            expected_sufficient_statistics,logZ = self.expected_sufficient_statistics(x,return_logZ=True)
            if loss_augmented:
                self.set_weights(w)
                logZ = logZ/alpha
                expected_sufficient_statistics = self.deaugment(expected_sufficient_statistics)
            f_s = np.dot(w,sufficient_statistics) - logZ
            f += f_s
            if return_gradient:
                avg_sufficient_statistics += sufficient_statistics - expected_sufficient_statistics
               
        # The log likelihood must be negative
        assert f < 0  
        
        if return_gradient:
            return f,avg_sufficient_statistics
        else:
            return f
            
    def get_mle_objective(self,X,Y,alpha=1.0,loss_augmented=False):
        # Precalculate all sufficient statistics
        if self.verbose: print "Precalculating Sufficient Statistics..."
        SS = []
        for x,y in zip(X,Y):
            SS.append(self.sufficient_statistics(x,y))
            
        def obj_fun_and_gradient(w,batch=range(len(SS))):
            self.set_weights(w)
            f,g = self.log_likelihood(X=X,Y=Y,SS=SS,return_gradient=True,alpha=alpha,loss_augmented=loss_augmented)
            
            # subtract l2 regularization
            g -= self.lambda_0*w
            f -= 0.5*self.lambda_0*(np.sum(w**2))
            
            # return negative regularized log likelihood and gradient
            return -f,-g
            
        return obj_fun_and_gradient
        
            
    def fit_mle(self,X,Y):
        regularized_nll_and_gradient = self.get_mle_objective(X,Y)
                
        # Use l-bfgs
        if self.method == "bfgs":
            w0 = np.zeros(self.n_parameters)
            res = minimize(regularized_nll_and_gradient,w0,jac=True,method='L-BFGS-B',options={'maxiter':self.max_iter,'disp':bool(self.verbose),"ftol":self.tol})
            self.set_weights(res.x)
            
        # use adagrad    
        elif self.method == "adagrad":
            w = np.zeros(self.n_parameters)
            accumulated_grad = np.zeros(w.shape)
            
            t = 0
            if self.track_history:
                self.history = []
                
            while t < self.max_iter and not converged:
                # choose a random batch of size batch_size with replacement
                batch = np.random.choice(len(SS),batch_size,replace=False)
                f,g = regularized_nll_and_gradient(w,batch)
                accumulated_grad += g**2
                w -= lr*g/(np.sqrt(accumulated_grad) + 0.000001)
                t += 1
                if self.track_history:
                    self.history.append(f)
                    
                if self.verbose > 1:
                    print "Iter:", t, "| Average regularized log likelihood:", self.history[-1]
                        
            self.set_weights(w)
    
    def initialize(self,X,Y):
        pass
            
    def fit_ssvm(self,X,Y):
        self.inference_calls = 0
        self.size_joint_feature = self.n_parameters
        ssvm_learner = NSlackSSVM(self,C=1.0/self.lambda_0,max_iter=self.max_iter,verbose=self.verbose)
        Y = [self.vectorize_label(y) for y in Y]
        ssvm_learner.fit(X,Y)
        self.set_weights(ssvm_learner.w)
        
    def fit_alpha_ssvm(self,X,Y):
        regularized_nll_and_gradient = self.get_mle_objective(X,Y)
                
        # Use l-bfgs
        if self.method == "bfgs":
            w = np.zeros(self.n_parameters)
            for alpha_exp in [10]:
                alpha = 10.0**alpha_exp
                print "Alpha =", alpha
                regularized_nll_and_gradient = self.get_mle_objective(X,Y,alpha=alpha,loss_augmented=True)
                # res = minimize(regularized_nll_and_gradient,w,jac=True,method='L-BFGS-B',options={'maxiter':self.max_iter,'disp':bool(self.verbose),"ftol":self.tol})
                res = minimize(regularized_nll_and_gradient,w,jac=True,method='L-BFGS-B',options={'maxiter':self.max_iter,'disp':0,"ftol":self.tol})
                print res.fun
                w = res.x
            self.set_weights(w)
            
        # use adagrad    
        elif self.method == "adagrad":
            raise NotImplementedError("fit_alpha_ssvm not implemented with adagrad.")
        
        