import unittest
import numpy as np
import torch
from pgcan import core
class ScientificContract(unittest.TestCase):
    def test_architecture_and_conservation(self):
        model=core.Model();self.assertEqual(sum(p.numel() for p in model.parameters()),5339)
        # This assertion exercises the softmax/accounting constraint, not a performance claim.
        p=torch.softmax(torch.tensor([-20.,0.,12.]),0).numpy()
        c=core.carbon(p)
        self.assertTrue(np.all(c>=0));self.assertAlmostEqual(float(c.sum()),15_000_000.,places=6)
    def test_original_graph_symmetry_and_self_loops(self):
        xy=np.array([(i,j) for i in range(4) for j in range(4)],float)*500
        a,*_=core.graph_for(xy,8);a=a.to_dense().numpy()
        np.testing.assert_allclose(a,a.T,atol=1e-7)
        self.assertTrue(np.all(np.diag(a)>0))
