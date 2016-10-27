from cfg_crf import *
import unittest

class TestCFGCRF(unittest.TestCase):
    
    def setUp(self):
        grammar =   "START\n\
                    a b c\n\
                    START -> A START2\n\
                    START2 -> B C\n\
                    A -> a A | a\n\
                    B -> b B | b\n\
                    C -> c C | c"
        
        with open("test_grammar.txt",'w') as f:
            f.write(grammar)
            
        self.mdl = CFGCRF("test_grammar.txt",1,1,objective='ssvm',lambda_0=1.0)
        # print self.mdl.rules_dict
        # print self.mdl.symbol_dict
        terminal_weights = np.array([0.0,0.0,1.0]).reshape((self.mdl.n_terminals,1))
        rule_weights = np.zeros((self.mdl.n_rules,1))
        self.mdl.set_weights(np.hstack((terminal_weights.flatten(),rule_weights.flatten())))

                
        self.y1 = [   'START',['A',['a',-1,-1],['A',['a',-1,-1],['A',['a',-1,-1],-1]]],\
                ['START2',['B',['b',-1,-1],['B',['b',-1,-1],-1]],\
                ['C',['c',-1,-1],-1]]]
                
        self.y2 = [   'START',['A',['a',-1,-1],['A',['a',-1,-1],-1]],\
                ['START2',['B',['b',-1,-1],['B',['b',-1,-1],-1]],\
                ['C',['c',-1,-1],['C',['c',-1,-1],-1]]]]
                
        n = 6
        x_terminal = np.ones(n,dtype=float).reshape((n,1))
        x_rule = []
        idx = 0
        for i in range(n):
            for j in range(i+1,n+1):
                for k in range(i+1,j+1):
                    x_rule.append(1.0)
        x_rule = np.array(x_rule).reshape((len(x_rule),1))
        self.x1 = [x_terminal,x_rule]
                    
    
    def test_load_grammar(self):
        expected_terminals = ['a','b','c']
        expected_non_terminals = ['START','START2','A','B','C']
        expected_grammar_dict = { 'START':[['A','START2']],\
                                'START2':[['B','C']],\
                                'A':[['a','A'],['a']],\
                                'B':[['b','B'],['b']],\
                                'C':[['c','C'],['c']]}
                                
        self.assertItemsEqual(self.mdl.terminals,expected_terminals)
        self.assertItemsEqual(self.mdl.non_terminals,expected_non_terminals)
        self.assertDictEqual(self.mdl.grammar_dict,expected_grammar_dict)
        
        for s in self.mdl.terminals:
            idx = self.mdl.symbol_dict[s]
            self.assertEqual(self.mdl.symbol_indices[idx],-1)
            
        for s in self.mdl.non_terminals:
            idx = self.mdl.symbol_dict[s]
            for p,production in zip(range(self.mdl.symbol_indices[idx],self.mdl.symbol_indices[idx+1]),self.mdl.grammar_dict[s]):
                self.assertEqual(self.mdl.rules[p,0],self.mdl.symbol_dict[production[0]])
                if len(production) == 2:
                    self.assertEqual(self.mdl.rules[p,1],self.mdl.symbol_dict[production[1]])
                else:
                    self.assertEqual(self.mdl.rules[p,1],-1)
              
                    
    # def test_set_weights(self):
    #     w = np.random.randn(self.mdl.n_parameters)
    #     self.mdl.set_weights(w)
    #     got = self.mdl.get_weight_vector()
    #
    #     self.assertTrue(np.all(w == got))
        
    def test_traversals(self):
        # expected_traversal = ['START','A','a','A','a','A','a','START2','B','b','B','b','C','c']
        expected_traversal = ['a','a','a','A','A','A','b','b','B','B','c','C','START2','START']
        expected_starts = [0,1,2,2,1,0,3,4,4,3,5,5,3,0]
        expected_ends   = [1,2,3,3,3,3,4,5,5,5,6,6,6,6]
        expected_splits = [1,2,3,3,2,1,4,5,5,4,6,6,5,3]
        expected_leaves = ['a','a','a','b','b','c']
        
        got_traversal,got_starts,got_ends,got_splits = zip(*[[node[0],i,j,k] for node,i,j,k in self.mdl.depth_first_traversal(self.y1,return_ij=True)])
        got_leaves = self.mdl.get_leaves(self.y1)
        
        self.assertListEqual(expected_traversal,list(got_traversal))
        self.assertListEqual(expected_starts,list(got_starts))
        self.assertListEqual(expected_ends,list(got_ends))
        self.assertListEqual(expected_splits,list(got_splits))
        self.assertListEqual(expected_leaves,list(got_leaves))
        
    def test_loss(self):
        y1_leaves = ['a','a','a','b','b','c']
        y2_leaves = ['a','a','b','b','c','c']

        self.assertListEqual(self.mdl.get_leaves(self.y1),y1_leaves)
        self.assertListEqual(self.mdl.get_leaves(self.y2),y2_leaves)
        self.assertEqual(self.mdl.loss(self.y1,self.y2),2)

    def test_feature_idx(self):
        n = 10
        idx = 0
        for i in range(n):
            for j in range(i+1,n+1):
                for k in range(i+1,j+1):
                    self.assertEqual(idx,self.mdl.get_feature_idx(n,i,j,k))
                    idx += 1

    def test_sufficient_statistics(self):
        expected_terminal_statistics = np.array([3.0,2.0,1.0])
        expected_rule_statistics = np.array([1,1,2,1,1,1,0,1],dtype=float)
        expected_statistics = np.hstack((expected_terminal_statistics,expected_rule_statistics))
        ss = self.mdl.sufficient_statistics(self.x1,self.y1)

        self.assertListEqual(list(ss),list(expected_statistics))

    def test_map_inference(self):
        y_hat,score = self.mdl._map_inference(self.x1,return_score=True)
        expected_traversal = ['a','A','b','B','c','c','c','c','C','C','C','C','START2','START']
        got_traversal = [node[0] for node in self.mdl.depth_first_traversal(y_hat)]
        self.assertListEqual(expected_traversal,got_traversal)
        jf = self.mdl.sufficient_statistics(self.x1,y_hat)
        self.assertEqual(np.dot(jf,self.mdl.get_weight_vector()),score)
        
        
    def sample_tree(self,symbol):
        if self.mdl.is_terminal(symbol):
            left = right = -1
        else:
            # expansion = np.random.choice(self.mdl.grammar_dict[symbol])
            expansion_idx = np.random.randint(0,len(self.mdl.grammar_dict[symbol]))
            expansion = self.mdl.grammar_dict[symbol][expansion_idx]
            
            left = self.sample_tree(expansion[0])
            if len(expansion) == 1:
                right = -1
            else:
                right = self.sample_tree(expansion[1])
                
        return [symbol,left,right]
        
    def sample_dataset(self,n_samples):
        X = []
        Y = []
        for s in range(n_samples):
            y = self.sample_tree("START")
            n = len(self.mdl.get_leaves(y))
            # print n,self.mdl.get_leaves(y)
            x0 = np.random.randn(n,1)
            x1_len = self.mdl.get_feature_idx(n,n-1,n,n) + 1
            x1 = np.random.randn(x1_len,1)
            Y.append(y)
            X.append([x0,x1])
            
        return X,Y
        
    def test_fit(self):
        # np.random.seed(1)
        n_samples = 25
        X,Y = self.sample_dataset(n_samples)
        self.mdl.fit(X,Y)
        
    def test_map_inference_2(self):
        np.random.seed(1)
        n_samples = 1
        X,Y = self.sample_dataset(n_samples)
        x,y = X[0],Y[0]
        w = np.random.randn(self.mdl.n_parameters)
        self.mdl.set_weights(w)
        # self.mdl.rule_weights[:,:] = 0.0
        
        y_hat,score = self.mdl._map_inference(x,return_score=True)
        jf_hat = self.mdl.sufficient_statistics(x,y_hat)
        jf = self.mdl.sufficient_statistics(x,y)
        self.assertAlmostEqual(np.dot(jf_hat,w),score)
        self.assertGreaterEqual(score,np.dot(jf,w))
        
if __name__=="__main__":
    unittest.main()
        