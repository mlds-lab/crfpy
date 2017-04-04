import numpy as np
from pystruct.learners import NSlackSSVM
import itertools as it
from scipy.misc import logsumexp
from scipy.optimize import minimize
# from pyhmc import hmc
# from rutils import keyboard

class CRF:
    """
        Base Conditional Random Field (CRF) class. The CRF class 
        implements maximum likelihood and max-margin learning (via
        pystruct). The user is expected to implement the appropriate
        inference algorithms for the desired model.
    """
    
    def __init__(self,lambda_0=0.0,max_iter=100,verbose=0,objective="mle",test_grad=False,method="bfgs",batch_size=100,lr=1.0,tol=1e-4):
        """
        Arguments:
            lambda_0 (float), default: 0.0: 
                Regularization constant.
            max_iter (int), default: 100:
                The maximum number of iterations to run the learning algorithm.
            verbose (int), default: 0:
                Verbosity (passed to optimization methods).
            objective {'mle','ssvm'}, default:'mle': 
                The objective to be optimized. 'mle' specifies that the
                parameters should be estimated by Maximum Likelihood Estimation
                and 'ssvm' specifies that the parameters should be estimated to 
                minimize the hinge loss.
            test_grad (Bool), default: False:
                For the 'mle' objective, this checks the gradient method against 
                the objective function using finite differences.
            method {'bfgs','adagrad'}, default:'bfgs':
                If using the 'mle' objective, this specifies the optimization algorithm
                to use.
            batch_size (int), default: 100:
                If using 'adagrad' to optimize the 'mle' objective, this specifies 
                mini-batch size.
            lr (float), default: 1.0:
                Learning rate for the 'adagrad' algorithm.
            tol (float), default: 1e-4:
                Tolerance for the all optimization algorithm.
        
        Returns:
            None
        
        """
        self.__dict__.update(locals())
        
    def fit(self,X,Y):
        """
        Estimates the parameters of the model for features X and labels Y.
    
        Arguments:
            X (list):
                List of features for each instance.
            Y (list):
                List of labels for each instance. These may be stored using any 
                structure with the exception that Y[i] may not be a tuple if 
                using the 'ssvm' objective (this is a restriction of PyStruct).
        Returns:
            None
        """
        if self.objective=="mle":
            self.fit_mle(X,Y)
        elif self.objective == "ssvm":
            self.fit_ssvm(X,Y)
        else:
            raise(ValueError("%s not implemented as a fit method."%self.objective))
        
    def sufficient_statistics(self, x, y):
        """
        Calculates the sufficient statistics vector for the instance pair (x,y).
        Necessary for all learning methods.
        
        Arguments:
            x: Instance features
            y: Instance labels
        
        Returns:
            phi (numpy.ndarray): 
                Sufficient statistics (or joint features) for the instance. Should 
                align with the parameter vector.
        """
        raise NotImplementedError( "sufficient_statistics not implemented." )
        
    def expected_sufficient_statistics(self, x, return_logZ=False):
        """
        Calculates the expected sufficient statistics vector (i.e. marginals) 
        for the instance pair (x,y).
        
        Arguments:
            x: 
                Instance features
            return_logZ (bool): 
                Indicates whether the log partition function should also be returned.
        
        Returns:
            phi_hat (numpy.ndarray):
                Expected sufficient statistics (or marginals) for the instance.
                Should align with the sufficient statistics vector.
            logZ (float):
                The log partition function. Returned only if return_logZ == True.
        """
        raise NotImplementedError( "expected_sufficient_statistics not implemented." )
        
    def get_weight_vector(self):
        """
        Get the current weights as a vector. Necessary for all learning methods.
        
        Returns:
            w (numpy.ndarray):
                Vector or weights.
        """
        raise NotImplementedError( "get_weight_vector not implemented." )
        
    def set_weights(self, w):
        """
        Takes a weight vector and seperates it for use in inference. For example, 
        a linear chain CRF would separate the weights in to unary potential and 
        transistion weights. Necessary for all learning methods.
        
        Arguments:
            w (numpy.ndarray):
                Vector or weights.
        
        Returns:
            None
        """
        raise NotImplementedError( "set_weights not implemented." )
        
    def map_inference(self, x, return_score=False):
        """
        Performs MAP inference. Necessary only for 'ssvm' learning, however, also 
        necessary for prediction.
        
        Arguments:
            x: 
                Instance features
            return_score (bool): 
                Indicates whether to return the unnormalized log probability of the
                MAP label.
        
        Returns:
            y_hat:
                MAP Label.
            score (float):
                The unnormalized log probability of the MAP label. Returned 
                only if return_score == True.
        """
        raise NotImplementedError("map_inference not implemented.")
        
        
    def set_loss_augmented_weights(self, x, y, w):
        """
        Augments the weights and features so that 
        w_aug^T phi(x_aug,y') = w^T phi(x,y') + loss(y,y'). This is possible 
        when the loss decomposes over the structure. Necessary for 'ssvm' learning.
        
        Arguments:
            x:
                Instance features.
            y: 
                Instance labels.
            w (numpy.ndarray):
                Vector or weights.
        
        Returns:
            x_aug:
                Augmented features.
        """
        raise NotImplementedError("set_loss_augmented_weights not implemented.")
        
    def loss(self,y,y_hat):
        """
        Calculates the loss between y and y_hat. Should be 
        semetric (i.e. loss(y,y') = loss(y',y)). Necessary for 'ssvm' learning. 
        
        Arguments:
            y: True label.
            y_hat: Predicted label.
        
        Return:
            loss (float): Loss.
        """
        raise NotImplementedError("loss not implemented.")
        
    def deaugment(self,w):
        raise NotImplementedError("deaugment not implemented.")
        
    def devectorize_label(self,y):
        return y
        
    def vectorize_label(self,y):
        return y
        
    def batch_map_inference(self,X, return_score=False):
        return [self.map_inference(x, return_score) for x in X]
        
    def joint_feature(self,x,y):
        y = self.devectorize_label(y)
        return self.sufficient_statistics(x,y)
        
    def inference(self,x,w):
        self.set_weights(w)
        return self.vectorize_label(self.map_inference(x))
        
    def loss_augmented_inference(self,x,y,w,relaxed=False):
        y = self.devectorize_label(y)
        self.inference_calls += 1
        x_aug = self.set_loss_augmented_weights(x,y,w)
        y_hat,score = self.map_inference(x_aug,return_score=True)
        # assert np.dot(self.joint_feature(x,y_hat),w) + self.loss(y,y_hat) == score
        if not np.isclose(np.dot(self.sufficient_statistics(x,y_hat),w) + self.loss(y,y_hat,vectorized_labels=False), score):
            # print "Inference Error: scores don't match"
            # w_aug = self.get_weight_vector()
            # jf_aug = self.joint_feature(x_aug,y_hat)
            # keyboard()
            raise ValueError("Inference Error: scores don't match")
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
        
    def predict(self,X):
        return [self.map_inference(x) for x in X]
        
    def get_params(self, deep=True):
        return {p:self.__dict__[p] for p in ["lambda_0","max_iter","verbose","objective","test_grad","method","batch_size","lr","tol"]}
        
        
        