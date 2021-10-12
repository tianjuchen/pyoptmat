import unittest

import numpy as np
import torch

from pyoptmat import temperature

class TestConstantParameter(unittest.TestCase):
  def test_value(self):
    pshould = torch.tensor([1.0,2.0])
    obj = temperature.ConstantParameter(pshould)
    pval = obj(torch.tensor(1.0))

    self.assertTrue(np.allclose(pshould.numpy(), pval.numpy()))

class TestPolynomialScaling(unittest.TestCase):
  def test_value(self):
    coefs = torch.tensor([1.1,2.5,3.0])
    x = torch.ones((100,))*1.51
    obj = temperature.PolynomialScaling(coefs)
    y1 = obj.value(x)

    y2 = np.polyval(coefs.numpy(), x)
    
    self.assertTrue(np.allclose(y1.numpy(), y2))

  def test_value_batch(self):
    coefs = torch.tensor([[1.1]*100,[2.5]*100,[3.0]*100])
    x = torch.ones((100,))*1.51
    obj = temperature.PolynomialScaling(coefs)
    y1 = obj.value(x)

    y2 = np.polyval(coefs.numpy(), x)
    
    self.assertTrue(np.allclose(y1.numpy(), y2))

  def test_value_constant(self):
    coefs = torch.tensor([2.51])
    x = torch.ones((100,))*1.51
    obj = temperature.PolynomialScaling(coefs)
    y1 = obj.value(x)

    y2 = np.polyval(coefs.numpy(), x)
    
    self.assertTrue(np.allclose(y1.numpy(), y2))

class TestKMRateSensitivityScaling(unittest.TestCase):
  def test_value(self):
    A = -8.679
    mu = temperature.PolynomialScaling(
        torch.tensor([-1.34689305e-02,-5.18806776e+00,7.86708330e+04]))
    b = 2.474e-7
    k = 1.38064e-20

    Ts = torch.linspace(25,950.0,50)+273.15
    
    obj = temperature.KMRateSensitivityScaling(A, mu, b, k)
    v1 = obj.value(Ts)

    mu_values = np.array([mu.value(T).numpy() for T in Ts])

    v2 = -mu_values*b**3.0/(k*Ts*A)

    self.assertTrue(np.allclose(v1.numpy(), v2))

  def test_value_batch(self):
    A = torch.linspace(-8.679-1,-8.679+1, 50)
    mu = temperature.PolynomialScaling(
        torch.tensor([-1.34689305e-02,-5.18806776e+00,7.86708330e+04]))
    b = 2.474e-7
    k = 1.38064e-20

    Ts = torch.linspace(25,950.0,50)+273.15
    
    obj = temperature.KMRateSensitivityScaling(A, mu, b, k)
    v1 = obj.value(Ts)

    mu_values = np.array([mu.value(T).numpy() for T in Ts])

    v2 = -mu_values*b**3.0/(k*Ts*A.numpy())

    self.assertTrue(np.allclose(v1.numpy(), v2))

class TestKMViscosityScaling(unittest.TestCase):
  def test_value(self):
    A = -8.679
    B = -0.744
    mu = temperature.PolynomialScaling(
        torch.tensor([-1.34689305e-02,-5.18806776e+00,7.86708330e+04]))
    b = 2.474e-7
    k = 1.38064e-20
    eps0 = 1e10

    Ts = torch.linspace(25,950.0,50)+273.15
    mu_values = np.array([mu.value(T).numpy() for T in Ts])

    obj = temperature.KMViscosityScaling(A, torch.tensor(B), mu, eps0, b, k)

    v1 = obj.value(Ts)
    v2 = np.exp(B)*mu_values*eps0**(k*Ts.numpy()*A/(mu_values*b**3.0))
    self.assertTrue(np.allclose(v1, v2))

  def test_value_batch(self):
    A = torch.linspace(-8.679-1,-8.679+1, 50)
    B = torch.linspace(-0.744,-0.80, 50)
    mu = temperature.PolynomialScaling(
        torch.tensor([-1.34689305e-02,-5.18806776e+00,7.86708330e+04]))
    b = 2.474e-7
    k = 1.38064e-20
    eps0 = 1e10

    Ts = torch.linspace(25,950.0,50)+273.15
    mu_values = np.array([mu.value(T).numpy() for T in Ts])

    obj = temperature.KMViscosityScaling(A, B, mu, eps0, b, k)

    v1 = obj.value(Ts)
    v2 = np.exp(B.numpy())*mu_values*eps0**(k*Ts.numpy()*A.numpy()/(mu_values*b**3.0))
    self.assertTrue(np.allclose(v1, v2))