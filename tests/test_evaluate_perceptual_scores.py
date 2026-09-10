import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from perceptual_scores import stratified_energy_loss


class PerceptualScoreTests(unittest.TestCase):
    def test_joint_distribution_matters_when_means_and_marginals_are_identical(self):
        record=dict(draw_counts=[2,2],component_weights=[.5,.5],omitted_mass=0.)
        same=np.array([[0.,0.],[0.,0.],[1.,1.],[1.,1.]])
        opposite=np.array([[0.,1.],[0.,1.],[1.,0.],[1.,0.]])
        np.testing.assert_array_equal(same.mean(axis=0),opposite.mean(axis=0))
        for target in ([0.,0.],[1.,1.]):
            correct=stratified_energy_loss(same,target,record)['energy_loss']
            incorrect=stratified_energy_loss(opposite,target,record)['energy_loss']
            self.assertAlmostEqual(correct,.25)
            self.assertAlmostEqual(incorrect,math.sqrt(.5)-.25)
            self.assertLess(correct,incorrect)

    def test_independent_pair_estimator_matches_exhaustive_discrete_expectation(self):
        record=dict(draw_counts=[2,2],component_weights=[.7,.2],omitted_mass=.1)
        target=.4
        # Independent two-point component laws give 16 equiprobable draw sets.
        losses=[]
        for a,b,c,d in itertools.product([0.,1.],repeat=4):
            values=np.array([[.2*a],[.2*b],[.6+.4*c],[.6+.4*d]])
            losses.append(stratified_energy_loss(values,[target],record)['energy_loss'])
        support=[0.,.2,.6,1.]
        mass=[.35,.35,.1,.1]
        expected=sum(w*abs(x-target) for w,x in zip(mass,support))
        expected-=.5*sum(w*v*abs(x-y) for w,x in zip(mass,support) for v,y in zip(mass,support))
        self.assertAlmostEqual(float(np.mean(losses)),expected,places=14)
        bernoulli=[stratified_energy_loss(np.array(x)[:,None],[0.],
                   dict(draw_counts=[2],component_weights=[1.],omitted_mass=0.))['energy_loss']
                   for x in itertools.product([0.,1.],repeat=2)]
        self.assertAlmostEqual(float(np.mean(bernoulli)),.25,places=14)

    def test_omission_bounds_keep_mass_and_numerical_radius_is_separate(self):
        record=dict(draw_counts=[4],component_weights=[.9],omitted_mass=.1)
        for target in (0.,.25,.5,1.):
            estimate=stratified_energy_loss(np.full((4,1),.2),[target],record)
            for missing in (0.,.5,1.):
                full=.9*abs(.2-target)+.1*abs(missing-target)-.09*abs(.2-missing)
                self.assertGreaterEqual(full+1e-15,estimate['energy_loss']+estimate['omission_lower_adjustment'])
                self.assertLessEqual(full-1e-15,estimate['energy_loss']+estimate['omission_upper_adjustment'])
        more=stratified_energy_loss(np.full((16,1),.2),[.5],dict(record,draw_counts=[16]))
        fewer=stratified_energy_loss(np.full((4,1),.2),[.5],record)
        self.assertAlmostEqual(more['numerical_radius'],fewer['numerical_radius']/2)
        expected=1.9*math.sqrt(.5*.9**2/4*math.log(40))
        self.assertAlmostEqual(fewer['numerical_radius'],expected)

    def test_small_nonzero_distances_survive_and_invalid_inputs_fail(self):
        record=dict(draw_counts=[2],component_weights=[1.],omitted_mass=0.)
        value=.5+1e-12
        result=stratified_energy_loss(np.full((2,3),value),np.full(3,.5),record)
        self.assertEqual(result['pair_dispersion_estimate'],0.)
        self.assertAlmostEqual(result['energy_loss'],value-.5,places=25)
        self.assertGreater(result['energy_loss'],0.)
        for bad in [dict(record,draw_counts=[1]),dict(record,draw_counts=[2.5]),
                    dict(record,component_weights=[.8]),dict(record,omitted_mass=-.1)]:
            with self.assertRaises(ValueError):
                stratified_energy_loss(np.zeros((2,1)),[0.],bad)
        with self.assertRaises(ValueError):
            stratified_energy_loss(np.zeros((2,1)),[1.1],record)


if __name__=='__main__':
    unittest.main()
