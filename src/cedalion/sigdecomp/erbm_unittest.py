import numpy as np  
import unittest 
from cedalion.sigdecomp.ERBM import simplified_ppval, cnstd_and_gain

class TestMathOperations(unittest.TestCase):

###### tests for simplified_ppval ########
    def test1_simplified_ppval(self): 
        table = np.load('measfunc_table.npy', allow_pickle= True)
        nf1  =  table[0]
        xs =  2.36716002  # point that is a break point   
        val_old = simplified_ppval(nf1['pp'], xs , version = 'old')    
        val_new = simplified_ppval(nf1['pp'], xs , version = 'new')  
        np.testing.assert_array_equal( val_old, val_new)  
    
    def test2_simplified_ppval(self): 
        table = np.load('measfunc_table.npy', allow_pickle= True)
        nf1  =  table[0]
        xs = 1.5  #  point that is not a break point   
        val_old = simplified_ppval(nf1['pp'], xs , version = 'old')    
        val_new = simplified_ppval(nf1['pp'], xs , version = 'new')  
        np.testing.assert_array_equal( val_old, val_new)   
    
    def test3_simplified_ppval(self): 
        table = np.load('measfunc_table.npy', allow_pickle= True)
        nf1  =  table[0]
        xs =  1.0 # point that is lower bound 
        val_old = simplified_ppval(nf1['pp'], xs , version = 'old')    
        val_new = simplified_ppval(nf1['pp'], xs , version = 'new')  
        np.testing.assert_array_equal( val_old, val_new)  
    
    def test4_simplified_ppval(self): 
        table = np.load('measfunc_table.npy', allow_pickle= True)
        nf1  =  table[0]
        xs =  3.0  # point that is upper bound 
        val_old = simplified_ppval(nf1['pp'], xs , version = 'old')    
        val_new = simplified_ppval(nf1['pp'], xs , version = 'new')  
        np.testing.assert_array_equal( val_old, val_new) 

    def test5_simplified_ppval(self): 
        table = np.load('measfunc_table.npy', allow_pickle= True)
        nf3  =  table[2]
        xs =  0.00107516
        val_old = simplified_ppval(nf3['pp'], xs , version = 'old')    
        val_new = simplified_ppval(nf3['pp'], xs , version = 'new')  
        np.testing.assert_array_equal( val_old, val_new) 

###### tests for cnstd_and_gain  ########
    def test1_cnstd_and_gain(self):
        p = 1 
        a = np.ones((p, 1 ))
        b_old, G_old =  cnstd_and_gain(a, version = 'old', test = True)    
        b_new, G_new =  cnstd_and_gain(a, version = 'new', test  = True) 
        np.testing.assert_array_equal(b_old, b_new)  
        np.testing.assert_array_equal(G_old, G_new)   

    def test2_cnstd_and_gain(self):
        np.random.seed(0)   
        p = 11
        a = np.random.rand(p, 1 )
        b_old, G_old =  cnstd_and_gain(a, version = 'old', test = True)    
        b_new, G_new =  cnstd_and_gain(a, version = 'new', test  = True) 
        np.testing.assert_array_equal(b_old, b_new)  
        np.testing.assert_array_equal(G_old, G_new)  
    
    def test3_cnstd_and_gain(self):  
        p = 11
        a = np.zeros((p, 1 ))
        b_old, G_old =  cnstd_and_gain(a, version = 'old', test = True)    
        b_new, G_new =  cnstd_and_gain(a, version = 'new', test  = True) 
        np.testing.assert_array_equal(b_old, b_new)  
        np.testing.assert_array_equal(G_old, G_new)  

    def test4_cnstd_and_gain(self):
        np.random.seed(0)   
        p = 20
        a = np.random.rand(p, 1 )
        b_old, G_old =  cnstd_and_gain(a, version = 'old', test = True)    
        b_new, G_new =  cnstd_and_gain(a, version = 'new', test  = True) 
        np.testing.assert_array_equal(b_old, b_new)  
        np.testing.assert_array_equal(G_old, G_new)  

    



if __name__ == '__main__':
    unittest.main()