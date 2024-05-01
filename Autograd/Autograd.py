import unittest
import numpy as np

def forward_f( i_x, i_y, i_z ):
   l_a = i_y + i_z
   l_b = i_x * l_a

   return l_b

def backward_f( i_x, i_y, i_z ):
  l_a = i_y + i_z

  l_dbda = i_x
  l_dbdx = l_a

  l_dady = 1
  l_dadz = 1

  l_dbdy = l_dbda * l_dady
  l_dbdz = l_dbda * l_dadz

  return l_dbdx, l_dbdy, l_dbdz

def forward_g(w0, w1, w2, x0, x1):
   l_a = w0*x0 + w1*x1 + w2
   l_b = np.exp(l_a)
   l_c = 1/(1+l_b)
   return l_c

def backward_g(w0, w1, w2, x0, x1):
   l_a = w0*x0 + w1*x1 + w2
   l_b = np.exp(l_a)
   l_c = 1/(1+l_b)

   l_dadx = w0
   l_dady = w1

   l_dbdx = l_b
   l_dbdy = l_b

   l_dcdx = (l_dadx*l_dbdx)/((1+l_dbdx)**2)
   l_dcdy = (l_dady*l_dbdy)/((1+l_dbdy)**2)

   return l_dcdx, l_dcdy

def forward_h(x,y):
   l_a = x*y
   l_b = x+y
   l_c = x-y
   l_d= (np.sin(l_a)+np.cos(l_b))/np.exp(l_c)

   return l_d

def backward_h(x,y):
   l_a = x*y
   l_b = x+y


   l_dadx = y
   l_dady = x

   l_dbdx = 1
   l_dbdy = 1
   
   l_dcdx = l_dadx*np.cos(l_a) + l_dbdx*np.sin(l_b)
   l_dcdy = l_dady*np.cos(l_a) + l_dbdy*np.sin(l_b)

   l_dedx = (l_dcdx + np.sin(l_a) + np.cos(l_b))/np.exp(x-y)
   l_dedy = (l_dcdy + np.sin(l_a) + np.cos(l_b))/np.exp(x-y)

   
   return


