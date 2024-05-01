import unittest
import Autograd

class AutogradTests(unittest.TestCase):
    def test_forward_f(self, x,y,z, target):
        self.assertAlmostEqual(Autograd.forward_f(x,y,z), target)
    
    def test_backward_f(self, x,y,z, target_x, target_y):
        self.assertAlmostEqual(Autograd.backward_f(x,y,z), target_x, target_y)

    def test_forward_g(self, w0,w1,w2, x, y, target):
        self.assertAlmostEqual(Autograd.forward_g(w0,w1,w2, x, y,), target)
    
    def test_backward_g(self, w0,w1,w2, x, y, target_x, target_y):
        self.assertAlmostEqual(Autograd.backward_g(w0,w1,w2, x, y,), target_x, target_y)

    def test_forward_h(self, x,y,z, target):
        self.assertAlmostEqual(Autograd.forward_h(x,y), target)
    
    def test_backward_h(self, x,y,z, target_x, target_y):
        self.assertAlmostEqual(Autograd.backward_h(x,y), target_x, target_y)