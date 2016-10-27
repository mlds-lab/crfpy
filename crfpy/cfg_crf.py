import numpy as np
from crf import CRF
from rutils import keyboard

class CFGCRF(CRF):
    def __init__(self,grammar_fn,n_terminal_features,n_rule_features,**kwargs):
        self.__dict__.update(locals())
        CRF.__init__(self,**kwargs)
        self.load_grammar(grammar_fn)
        self.n_parameters = self.get_n_parameters()
        
    def get_n_parameters(self):
        return self.n_terminal_features*len(self.terminals) + self.n_rule_features*len(self.rules)
        
    def load_grammar(self,grammar_fn):
        # Read in grammar file
        with open(grammar_fn, "r") as file:
            lines = file.readlines()
            
        # Get start symbol
        self.start_symbol = lines[0].rstrip('\n')
        
        # Get terminal symbols
        self.terminals = lines[1].rstrip('\n').split() # list of terminal symbols
        # self.terminals = dict(zip(terminal_list,range(len(terminal_list))))
        
        self.grammar_dict = {} # maps non-terminal symbols to lists of productions
        self.non_terminals = [] # list of non-terminal symbols
        # for each line in the input file
        for line in lines[2:]:
            # split the symbol and all posible productions
            lhs, rhs = line.rstrip('\n').split(' -> ') 
            lhs = lhs.replace(' ', '')
            assert lhs not in self.grammar_dict
            
            # Parse possible productions
            self.grammar_dict[lhs] = []
            self.non_terminals.append(lhs)
            for targets in rhs.split(' | '):
                self.grammar_dict[lhs].append(targets.split(" "))
                
        # Dictionary mapping symbols to an index
        self.symbol_dict = dict(zip(self.terminals+self.non_terminals,range(len(self.terminals)+len(self.non_terminals))))
        self.r_symbol_dict = dict(zip(self.symbol_dict.values(),self.symbol_dict.keys()))
        
        # Array mapping symbols to a range of rows in the rules array 
        # (a la indptr in scipy.sparse.csr_matrix)
        self.symbol_indices = np.zeros(len(self.terminals)+len(self.non_terminals)+1,dtype=int)
        # The index for all terminal symbols is -1
        self.symbol_indices[:len(self.terminals)] = -1
        
        # Create an array of productions ordered by source
        self.rules = []
        self.rules_dict = {}
        cur_idx = 0
        for i,symbol in enumerate(self.non_terminals):
            self.symbol_indices[len(self.terminals) + i] = cur_idx
            for expansion in self.grammar_dict[symbol]:
                left = self.symbol_dict[expansion[0]]
                if len(expansion) == 1:                    
                    right = -1
                    right_symbol = -1
                else:
                    right = self.symbol_dict[expansion[1]]
                    right_symbol = expansion[1]
                self.rules.append([left,right])
                self.rules_dict[(symbol,(expansion[0],right_symbol))] = cur_idx
                cur_idx += 1
                    
        self.symbol_indices[-1] = len(self.rules)
        self.rules = np.array(self.rules)
        self.n_terminals = len(self.terminals)
        self.n_rules = len(self.rules)
        
    def get_feature_idx(self,n,i,j,k):
        idx = i*(i**2-3*i*(n+1)+3*(n)**2+6*(n)+2)/6 
        idx += (i-j)*(i-j+1)/2
        idx += k - i - 1
        return idx
        
    def depth_first_traversal(self,y,return_ij=False):
        stack = [[y,0,0,False,False]]
        offset = 0
        while len(stack) > 0:
            node,i,k,traversed_left,traversed_right = stack[-1]
            if node == -1:
                stack.pop()
                continue
            elif self.symbol_dict[node[0]] < self.n_terminals:
                stack.pop()
                offset += 1
                if return_ij:
                    yield node,i,offset,offset
                else:
                    yield node
            elif not traversed_left:
                stack[-1][3] = True
                stack.append([node[1],offset,offset,False,False])
            elif not traversed_right:
                stack[-1][4] = True
                stack[-1][2] = offset
                stack.append([node[2],offset,offset,False,False])
            else:
                stack.pop()
                if return_ij:
                    yield node,i,offset,k
                else:
                    yield node
    
    def is_terminal(self,symbol):
        return self.symbol_dict[symbol] < self.n_terminals
    
    def get_rule(self,node):
        if node[1] == -1:
            return -1
        elif node[2] == -1:
            return self.rules_dict[(node[0],(node[1][0],-1))]
        else:
            return self.rules_dict[(node[0],(node[1][0],node[2][0]))]
            
    def get_leaves(self,y):
        leaves = []
        for node in self.depth_first_traversal(y):
            if self.is_terminal(node[0]):
                leaves.append(node[0])
                
        return leaves
        
    def sufficient_statistics(self, x, y):
        n = x[0].shape[0]
        terminal_statistics = np.zeros((len(self.terminals),x[0].shape[1]))
        rule_statistics = np.zeros((len(self.rules),x[1].shape[1]))
        for node,i,j,k in self.depth_first_traversal(y,return_ij=True):
            if self.is_terminal(node[0]):
                terminal_statistics[self.symbol_dict[node[0]]] += x[0][i]
            else:
                feature_idx = self.get_feature_idx(n,i,j,k)
                rule = self.get_rule(node)
                rule_statistics[rule] += x[1][feature_idx]
        
        return np.hstack((terminal_statistics.flatten(),rule_statistics.flatten()))
        
    def get_weight_vector(self):
        return np.hstack((self.terminal_weights.flatten(),self.rule_weights.flatten()))
        
    def set_weights(self, w):
        self.terminal_weights = w[:self.n_terminal_features*len(self.terminals)].reshape((len(self.terminals),-1))
        self.rule_weights = w[self.n_terminal_features*len(self.terminals):].reshape((len(self.rules),-1))
        
    def convert_input_parse_tree(self,y):
        y_out = self.build_tree_recursive(y)
        self.propogate_lengths(y_out,0)
        return y_out
        
    def propogate_lengths_recursive(self,y,offset):
        if y == -1:
            return 0
            
        y[1] += offset
        y[2] += offset
        offset = self.propogate_lengths_recursive(y[3],offset)
        offset = self.propogate_lengths_recursive(y[4],offset)
        return y[2]
        
    def build_tree_recursive(self,y):
        if self.symbol_dict[y[0]] < self.n_terminals:
            return [y[0],0,1,-1,-1]
            
        if y[1] != -1:
            left = self.convert_parse_tree(y[1])
        else:
            left = -1
            
        if y[2] != -1:
            right = self.convert_parse_tree(y[2])
        else:
            right = -1
            
        return [y[0],0,left[2]+right[2],left,right]
    
    def map_inference(self, X, return_score=False):
        return [self._map_inference(x,return_score) for x in X]
        
    def _map_inference(self, x, return_score=False):
        n = x[0].shape[0]
        stack = [(self.symbol_dict[self.start_symbol],0,n)]
        scores = {}
        trace = {}
        
        idx = 0
        while len(stack) > 0:
            idx += 1
            symbol,i,j = stack[-1]
            # print symbol,i,j
            if (symbol,i,j) in scores:
                stack.pop()
                continue
            
            if symbol < self.n_terminals:
                # print symbol,i,j
                if i == j-1:
                    scores[(symbol,i,j)] = np.dot(self.terminal_weights[symbol],x[0][i])
                    # print "score:",scores[(symbol,i,j)]
                else:
                    scores[(symbol,i,j)] = -np.inf
                stack.pop()
                continue
            
            best_score = -np.inf
            best_rule = None
            best_expansion = None
            all_children_cached = True
            
            idx0,idx1 = self.symbol_indices[symbol],self.symbol_indices[symbol+1]
            # for rule,expansion in zip(range(idx0,idx1),self.rules[idx0,idx1]):
            for rule,expansion in zip(range(idx0,idx1),self.rules[idx0:idx1]):
                left,right = expansion
                if i == j: continue
                if right == -1:
                    krange = [j]
                elif left < len(self.terminals):
                    krange = [i+1]
                elif right < len(self.terminals):
                    krange = [j-1]
                else:
                    krange = range(i+1,j)
                    
                for k in krange:
                    if (left,i,k) not in scores:
                        stack.append((left,i,k))
                        all_children_cached = False
                    if (right,k,j) not in scores and right != -1:
                        stack.append((right,k,j))
                        all_children_cached = False
                    if not all_children_cached: continue
                    
                    score = scores[(left,i,k)]
                    if right != -1:
                        score += scores[(right,k,j)]
                    feature_idx = self.get_feature_idx(n,i,j,k)
                    score += np.dot(self.rule_weights[rule],x[1][feature_idx])
                    
                    # print symbol,i,j,k,left,right,score
                    
                    if score > best_score:
                        best_score = score
                        best_rule = (k,left,right)
                        
            if all_children_cached:
                scores[(symbol,i,j)] = best_score
                trace[(symbol,i,j)] = best_rule
               
        # k,left,right = trace[(self.symbol_dict[self.start],0,n)]
        # y_hat = Node(self.symbol_dict[self.start],0,n)
        # y_hat.build_from_trace(trace)
        # keyboard()
        y_hat = self.build_label_from_trace(trace,self.symbol_dict[self.start_symbol],0,n)
        assert len(self.get_leaves(y_hat)) == n
        if return_score:
            return y_hat,scores[(self.symbol_dict[self.start_symbol],0,n)]
        else:
            return y_hat
        
    def build_label_from_trace(self,trace,symbol_idx,i,j):
        if symbol_idx == -1:
            return -1

        symbol = self.r_symbol_dict[symbol_idx]
        # print symbol,i,j
        if symbol_idx < self.n_terminals:
            return [symbol,-1,-1]
        else:
            k,left,right = trace[(symbol_idx,i,j)]
            left_tree = self.build_label_from_trace(trace,left,i,k)
            right_tree = self.build_label_from_trace(trace,right,k,j)
            return [symbol,left_tree,right_tree]
        
    def set_loss_augmented_weights(self, x, y, w):
        self.set_weights(w)
        n = x[0].shape[0]
        
        w0_class = np.eye(len(self.terminals))
        self.terminal_weights = np.hstack((self.terminal_weights,w0_class))

        x0_class = np.ones((n,self.n_terminals))
        y0 = np.array([self.symbol_dict[s] for s in self.get_leaves(y)])
        x0_class[range(x0_class.shape[0]),y0] = 0.0
        x0_aug = np.hstack((x[0],x0_class))
        
        return [x0_aug,x[1]]
        
    def loss(self,y,y_hat,vectorized_labels=False):
        y0 = np.array(self.get_leaves(y))
        y0_hat = np.array(self.get_leaves(y_hat))
        return np.sum(y0 != y0_hat)
        
    def deaugment(self,w):
        offset = (self.n_terminal_features+len(self.terminals))*len(self.terminals)
        terminal_weights = w[:offset].reshape((len(self.terminals),-1))
        terminal_weights = terminal_weights[:,:self.n_terminal_features]
        rule_weights = w[offset:].reshape((len(self.rules),-1))
        
        return np.hstack((terminal_weights.flatten(),rule_weights.flatten()))
        
        
        
    