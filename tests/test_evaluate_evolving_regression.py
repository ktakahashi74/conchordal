import copy
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evolving_regression import EvolvingResidualRegression
from residual_regression import ResidualRegression, cdf, log_density


class EvolvingRegressionTests(unittest.TestCase):
    def test_irregular_time_filter_matches_full_signed_batch_covariance(self):
        x = np.array([[1., .2], [1., -.5], [.3, .7], [.2, -.1]])
        times, offsets, values = np.array([.1, .3, .9, 1.4]), np.array([.1, .4, .2, .3]), np.array([.7, .3, .2, .6])
        m0, v0, a0, b0 = np.array([.2, -.1]), np.array([[.4, .05], [.05, .3]]), 3., .04
        for rate in (.2, 2.):
            model = EvolvingResidualRegression(m0, v0, drift_per_sec=rate, initial_time_sec=0.,
                                               prior_shape=a0, prior_residual_variance=.02, max_paths=4096)
            previous = 0.
            for i, y in enumerate(values):
                forecast = model.forecast(times[i], x[i], offset=offsets[i])
                update = model.observe(times[i], x[i], y, offset=offsets[i])
                n = i+1; design=x[:n]; t=times[:n]
                response=np.array(list(itertools.product((-1.,1.),repeat=n)))*values[:n]-offsets[:n]
                centered=response-design@m0
                kernel=np.eye(n)+design@v0@design.T+rate*np.minimum(t[:,None],t)
                solved=np.linalg.solve(kernel,centered.T)
                bn=b0+.5*np.einsum('ij,ji->i',centered,solved)
                evidence=(gammaln(a0+n/2)-gammaln(a0)-n/2*math.log(2*math.pi)
                          -.5*np.linalg.slogdet(kernel)[1]+a0*math.log(b0)-(a0+n/2)*np.log(bn))
                total=float(logsumexp(evidence));order=np.argsort(-evidence,kind='stable')
                self.assertAlmostEqual(log_density(forecast,y),total-previous,places=11)
                self.assertAlmostEqual(update['filter_log_density'],total-previous,places=11)
                cross=np.vstack((v0@design.T,rate*t))
                covariance=np.zeros((3,3));covariance[:2,:2]=v0;covariance[-1,-1]=rate*t[-1]
                covariance-=cross@np.linalg.solve(kernel,cross.T)
                means=np.r_[m0,0.]+(cross@solved).T
                np.testing.assert_allclose(model.posterior.mean,means[order],atol=1e-13)
                np.testing.assert_allclose(model.posterior.relative_covariance,covariance,atol=1e-13)
                np.testing.assert_allclose(model.posterior.scale,bn[order],atol=1e-13)
                np.testing.assert_allclose(model.posterior.log_weights,(evidence-total)[order],atol=1e-12)
                previous=total
            self.assertGreater(np.max(abs(model.posterior.relative_covariance[:2,-1])),.01)

    def test_zero_diffusion_preserves_stationary_forecasts_and_updates_exactly(self):
        rng=np.random.default_rng(192073)
        for size in (0,3):
            kwargs=dict(prior_shape=3.,prior_residual_variance=.02,max_paths=128)
            plain=ResidualRegression(np.zeros(size),.25*np.eye(size),**kwargs)
            timed=EvolvingResidualRegression(np.zeros(size),.25*np.eye(size),drift_per_sec=0.,
                                            initial_time_sec=0.,**kwargs)
            for i in range(1,21):
                x=rng.normal(size=size);offset=float(rng.uniform(0,1));y=0. if i%3==0 else float(rng.uniform(0,1))
                a,b=plain.forecast(x,offset=offset),timed.forecast(i*.2,x,offset=offset)
                for key in a:np.testing.assert_array_equal(a[key],b[key])
                self.assertEqual(plain.observe(x,y,offset=offset),timed.observe(i*.2,x,y,offset=offset))

    def test_projection_preserves_memory_marginal_and_frozen_predictions(self):
        model=EvolvingResidualRegression([0.,0.],.25*np.eye(2),drift_per_sec=1.,initial_time_sec=0.,
                                        prior_shape=3.,prior_residual_variance=.02,max_paths=128)
        model.observe(.2,[.8,.6],.5,offset=.3)
        f=model.forecast(1.,[.6,.8],offset=.4);saved=copy.deepcopy(f)
        before=copy.deepcopy(model.posterior)
        model.advance_to(.5);model.advance_to(.8)
        np.testing.assert_array_equal(model.posterior.mean[:,:-1],before.mean[:,:-1])
        np.testing.assert_array_equal(model.posterior.relative_covariance[:-1,:-1],before.relative_covariance[:-1,:-1])
        np.testing.assert_array_equal(model.posterior.scale,before.scale)
        np.testing.assert_array_equal(model.posterior.log_weights,before.log_weights)
        self.assertEqual(model.last_outcome_time_sec,.2)
        g=model.forecast(1.,[.6,.8],offset=.4)
        np.testing.assert_allclose(cdf(f,[0,.3,1,3]),cdf(g,[0,.3,1,3]),atol=1e-14)
        np.testing.assert_allclose(g['signed_component_variance'],
            g['residual_variance_mean']+g['within_component_parameter_variance']+
            g['within_component_state_variance']+2*g['within_component_parameter_state_covariance'],atol=1e-14)
        model.observe(1.,[.6,.8],.4,offset=.4)
        for key in f:np.testing.assert_array_equal(f[key],saved[key])

    def test_time_units_zero_continuity_and_invalid_input_preserve_state(self):
        kwargs=dict(prior_shape=3.,prior_residual_variance=.02,max_paths=128)
        a=EvolvingResidualRegression([0.],[[.25]],drift_per_sec=.7,initial_time_sec=0.,**kwargs)
        b=EvolvingResidualRegression([0.],[[.25]],drift_per_sec=.7/1000,initial_time_sec=0.,**kwargs)
        for i,y in enumerate((.3,.5,.2),1):
            a.observe(.2*i,[1.],y,offset=.4);b.observe(200*i,[1.],y,offset=.4)
        np.testing.assert_allclose(cdf(a.forecast(1.,[1.],offset=.4),[0,.2,1]),
                                   cdf(b.forecast(1000.,[1.],offset=.4),[0,.2,1]),atol=1e-14)
        zero,tiny=copy.deepcopy(a),copy.deepcopy(a)
        zero.observe(.8,[1.],0.,offset=.4);tiny.observe(.8,[1.],1e-24,offset=.4)
        np.testing.assert_allclose(cdf(zero.forecast(1.,[1.],offset=.4),[0,.2,1]),
                                   cdf(tiny.forecast(1.,[1.],offset=.4),[0,.2,1]),atol=1e-14)
        before=copy.deepcopy(a.posterior)
        with self.assertRaises(ValueError):a.observe(.8,[1.],float('nan'),offset=.4)
        np.testing.assert_array_equal(a.posterior.relative_covariance,before.relative_covariance)
        self.assertAlmostEqual(a.clock_sec,.6)
        with self.assertRaises(ValueError):a.forecast(.5,[1.],offset=.4)


if __name__=='__main__':unittest.main()
