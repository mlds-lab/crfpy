from crf import CRF
import numpy as np
from alpha_beta import *

class LinearChainCRF(CRF):
    def __init__(self,n_classes,n_features,**kwargs):
        self.__dict__.update(locals())
        CRF.__init__(self,**kwargs)
        self.n_classes = int(self.n_classes)
        self.n_features = int(self.n_features)
        self.n_parameters = int(self.n_classes*self.n_features + self.n_classes**2)
        
    def forward(self,x,phi=None):
        N = x.shape[0]
        if phi is None:
            unary_potentials = x.dot(self.feature_weights.T)
        else:
            unary_potentials = phi
        
        alpha = cy_alpha(unary_potentials,self.transition_weights)
        return alpha
        
    def backward(self,x,phi=None):
        N = x.shape[0]
        if phi is None:
            unary_potentials = x.dot(self.feature_weights.T)
        else:
            unary_potentials = phi
        beta = cy_beta(unary_potentials,self.transition_weights)
        return beta
        
    def sufficient_statistics(self,x,y):
        N = x.shape[0]
        feature_statistics = np.zeros((self.n_classes,self.n_features))
        for i in range(N):
            feature_statistics[y[i]] += x[i]
            
        transition_statistics = np.zeros((self.n_classes,self.n_classes))
        for i in range(N-1):
            transition_statistics[y[i],y[i+1]] += 1.0
            
        sufficient_statistics = np.hstack((feature_statistics.flatten(),transition_statistics.flatten()))
        return sufficient_statistics
        
    def expected_sufficient_statistics(self,x,return_logZ=False):
        N = x.shape[0]
        unary_potentials = x.dot(self.feature_weights.T)
        alpha = self.forward(x,unary_potentials)
        beta = self.backward(x,unary_potentials)
        
        logZ = logsumexp(alpha[N-1])
        unary_marginals = np.clip(np.exp(alpha + beta - logZ),0.0,1.0)
        expected_feature_statistics = (x.T.dot(unary_marginals)).T
            
        pairwise_marginals = np.clip(np.exp(    alpha[:N-1].reshape((N-1,self.n_classes,1)) 
                                                + (unary_potentials[1:] + beta[1:]).reshape((N-1,1,self.n_classes)) 
                                                + self.transition_weights.reshape((1,self.n_classes,self.n_classes)) 
                                                - logZ),0.0,1.0)
        expected_transition_statistics = np.sum(pairwise_marginals,0)
            
        expected_sufficient_statistics = np.hstack((expected_feature_statistics.flatten(),expected_transition_statistics.flatten()))
        if return_logZ:
            return expected_sufficient_statistics,logZ
        else:
            return expected_sufficient_statistics
        
    def set_weights(self,w):
        self.transition_weights = w[self.n_classes*self.n_features:].reshape((self.n_classes,self.n_classes))
        self.feature_weights = w[:self.n_classes*self.n_features].reshape((self.n_classes,self.n_features))
        
        
    def get_weight_vector(self):
        return np.hstack((self.feature_weights.flatten(),self.transition_weights.flatten()))

    def get_params(self, deep=True):
        return CRF.get_params(self,deep)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
        
    def map_inference(self,X,return_score=False):
        return [self.viterbi(x,return_score=return_score) for x in X]
    
    def viterbi(self,x,return_score=False):
        # init
        n = x.shape[0]
        alpha = np.zeros((n,self.n_classes)) # dp values
        alpha_path = np.zeros((n,self.n_classes),dtype=int) # dp path
        
        # calculate node potentials
        unary_potentials = x.dot(self.feature_weights.T)
        
        # viterbi
        alpha[0] = unary_potentials[0]
        for i in range(1,n):
            local_transition_matrix = self.transition_weights.T + alpha[i-1].reshape((1,self.n_classes))
            alpha_path[i] = np.argmax(local_transition_matrix,1)
            alpha[i] = unary_potentials[i] + local_transition_matrix[range(self.n_classes),alpha_path[i]]
            
        # back out labeling
        Y = np.zeros(n)
        Y[n-1] = np.argmax(alpha[n-1])
        if return_score:
            score = np.max(alpha[n-1])
        
        for i in range(n-1)[::-1]:
            Y[i] = alpha_path[i+1,Y[i+1]]
            
        if return_score:
            return Y,score
        else:
            return Y
        
    def set_loss_augmented_weights(self,x,y,w,alpha=1.0):
        self.set_weights(alpha*w)
        self.feature_weights = np.hstack((self.feature_weights,alpha*np.eye(self.n_classes)))
        
        n = x.shape[0]
        x_aug = np.ones((n,self.n_classes))
        x_aug[range(n),y.astype(int)] = 0.0
        x_aug = np.hstack((x,x_aug))
        
        return x_aug
        
    def loss(self,y,y_hat):
        return np.sum(y!=y_hat)
        
    def deaugment(self,w):
        transition_weights = w[self.n_classes*(self.n_features+self.n_classes):].reshape((self.n_classes,self.n_classes))
        feature_weights = w[:self.n_classes*(self.n_features+self.n_classes)].reshape((self.n_classes,(self.n_features+self.n_classes)))
        feature_weigths = feature_weights[:,:self.n_features]
        
        return np.hstack((feature_weigths.flatten(),transition_weights.flatten()))
        
        
def test_mle_learning():
    data = np.load("../data/test_data/linear_chain_data.npy")[0]
    X,Y = zip(*data)
    mdl = LinearChainCRF(2,2,lambda_0=1.0,verbose=1)
    # keyboard()
    mdl.fit(X,Y)
    print mdl.transition_weights
    print mdl.feature_weights
    
def test_ssvm_learning():
    data = np.load("../data/test_data/linear_chain_data.npy")[0]
    X,Y = zip(*data)
    mdl = LinearChainCRF(2,2,lambda_0=1.0,verbose=1,objective="ssvm")
    # keyboard()
    mdl.fit(X,Y)
    print mdl.transition_weights
    print mdl.feature_weights
    
def test_alpha_ssvm_learning():
    data = np.load("../data/test_data/linear_chain_data.npy")[0]
    X,Y = zip(*data)
    mdl = LinearChainCRF(2,2,lambda_0=1.0,verbose=1,objective="alpha_ssvm")
    # keyboard()
    mdl.fit(X,Y)
    print mdl.transition_weights
    print mdl.feature_weights
    
if __name__=="__main__":
    test_mle_learning()
    test_ssvm_learning()
    test_alpha_ssvm_learning()
    
        