import copy
from functools import lru_cache
import itertools
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uncertain_contexts import MagnitudeContexts
from uncertain_regression import MagnitudeRegression,cdf,log_density


class UncertainContextTests(unittest.TestCase):
    def test_context_and_sign_quotient_matches_exhaustive_batch_evidence(self):
        design=np.array([[.4],[-.2],[.6],[.1]])
        values=np.array([.7,0.,1.2,.4])
        for mode,contexts in [('stationary',2),('renewal',2),('recurrent',2),('recurrent',3)]:
            model=MagnitudeContexts(1,mode=mode,contexts=contexts,residual_sd=.2,
                                    observation_sd=.3,max_paths=4096,change_rate_hz=.4)
            @lru_cache(None)
            def block(indices):
                if not indices:return 0.
                x=np.column_stack([np.ones(len(indices)),design[list(indices),0]])
                y=values[list(indices)]
                covariance=.13*np.eye(len(indices))+x@np.diag([100.,1.])@x.T
                logdet=np.linalg.slogdet(covariance)[1]
                terms=[]
                for signs in itertools.product([-1.,1.],repeat=len(indices)):
                    z=y*signs
                    terms.append(-.5*(len(indices)*math.log(2*math.pi)+logdet
                                      +z@np.linalg.solve(covariance,z)))
                return float(logsumexp(terms))
            previous=0.
            for i,value in enumerate(values):
                forecast=model.forecast((i+1)*.2,design[i])
                result=model.observe((i+1)*.2,design[i],value)
                scores=[]
                if mode=='recurrent':
                    decay=math.exp(-.4*contexts/(contexts-1)*.2)
                    stay=1/contexts+(1-1/contexts)*decay
                    other=(1-decay)/contexts
                    for assignment in itertools.product(range(contexts),repeat=i+1):
                        states=np.array(assignment)
                        score=-math.log(contexts)+sum(math.log(stay if a==b else other)
                                                     for a,b in zip(states[:-1],states[1:]))
                        score+=sum(block(tuple(np.flatnonzero(states==k))) for k in range(contexts))
                        scores.append(score)
                else:
                    partitions=(itertools.product((0,1),repeat=i) if mode=='renewal'
                                else [tuple(0 for _ in range(i))])
                    for bits in partitions:
                        bounds=[0,*[j+1 for j,bit in enumerate(bits) if bit],i+1]
                        score=(sum(bits)*math.log(-math.expm1(-.08))-(i-sum(bits))*.08
                               if mode=='renewal' else 0.)
                        scores.append(score+sum(block(tuple(range(a,b))) for a,b in zip(bounds[:-1],bounds[1:])))
                total=float(logsumexp(scores))
                self.assertAlmostEqual(result['filter_log_density'],total-previous,places=10)
                self.assertAlmostEqual(log_density(forecast,value),total-previous,places=10)
                self.assertLess(result['pruned_mass'],1e-12)
                previous=total

    def test_stationary_predictions_match_the_unquotiented_regressor(self):
        context=MagnitudeContexts(1,mode='stationary',residual_sd=.1,observation_sd=.2,max_paths=512)
        reference=MagnitudeRegression([0.,0.],np.diag([100.,1.]),residual_sd=.1,max_paths=512)
        for i,value in enumerate([1.,0.,.3,1.4,.8,1e-12,2.]):
            x=[math.sin(i*.7)]
            a=context.observe(i+1.,x,value)
            b=reference.observe([1.,*x],value,observation_sd=.2)
            self.assertAlmostEqual(a['filter_log_density'],b['filter_log_density'],places=9)
            np.testing.assert_allclose(cdf(context.forecast(i+1.5,x),[0.,.2,1.,2.]),
                                       cdf(reference.forecast([1.,*x],observation_sd=.2),[0.,.2,1.,2.]),atol=1e-10)

    def test_zero_update_continuity_and_sign_symmetry(self):
        for mode in ('stationary','renewal','recurrent'):
            model=MagnitudeContexts(1,mode=mode,residual_sd=.1,observation_sd=.1,max_paths=4096)
            for i in range(3):model.observe(i+1.,[.3*i],1.)
            zero=copy.deepcopy(model)
            zero.observe(4.,[.2],0.)
            points=[0.,.2,.5,1.,2.]
            for value in [1e-8,1e-24,1e-48]:
                branch=copy.deepcopy(model)
                result=branch.observe(4.,[.2],value)
                self.assertLess(result['pruned_mass'],1e-12)
                np.testing.assert_allclose(cdf(branch.forecast(5.,[.7]),points),
                                           cdf(zero.forecast(5.,[.7]),points),atol=1e-10)
            flipped=copy.deepcopy(model)
            flipped.information[:,0]*=-1
            np.testing.assert_allclose(cdf(flipped.forecast(4.,[.2]),points),
                                       cdf(model.forecast(4.,[.2]),points),atol=1e-14)

    def test_missing_time_and_frozen_issuance(self):
        model=MagnitudeContexts(0,mode='recurrent',residual_sd=.1,observation_sd=.1,max_paths=128)
        model.observe(1.,[],1.)
        issued=model.forecast(4.,[])
        saved=copy.deepcopy(issued)
        before=copy.deepcopy(model)
        model.skip_to(2.)
        for name in ('cov','information','used','active','log_weights'):
            np.testing.assert_array_equal(getattr(model,name),getattr(before,name))
        np.testing.assert_allclose(model.transition(.3)@model.transition(1.7),model.transition(2.),atol=1e-15)
        for name in issued:np.testing.assert_array_equal(model.forecast(4.,[])[name],saved[name])
        model.observe(3.,[],0.)
        self.assertEqual(model.completed,2)
        for name in issued:np.testing.assert_array_equal(issued[name],saved[name])
        with self.assertRaises(ValueError):model.forecast(3.,[])

    def test_inactive_parameters_survive_and_return_without_forgetting(self):
        model=MagnitudeContexts(0,mode='recurrent',residual_sd=.01,observation_sd=.01,max_paths=1)
        for i in range(20):model.observe((i+1)*.1,[],1.)
        model.observe(2.1,[],3.)
        self.assertEqual(model.active[0],1)
        cov,information=model.cov[0,0].copy(),model.information[0,0].copy()
        for i in range(22,31):model.observe(i*.1,[],3.)
        np.testing.assert_array_equal(model.cov[0,0],cov)
        np.testing.assert_array_equal(model.information[0,0],information)
        result=model.observe(3.1,[],1.)
        self.assertEqual(model.active[0],0)
        self.assertGreater(result['returning_context_mass'],.9)

    def test_pruning_mass_is_measured_after_equivalent_states_merge(self):
        full=MagnitudeContexts(0,mode='recurrent',residual_sd=.1,observation_sd=.1,max_paths=128)
        small=MagnitudeContexts(0,mode='recurrent',residual_sd=.1,observation_sd=.1,max_paths=1)
        a=full.observe(1.,[],1.)
        b=small.observe(1.,[],1.)
        self.assertEqual(a['retained_paths'],1)
        self.assertAlmostEqual(b['pruned_mass'],0.)
        full.observe(2.,[],1.5)
        result=small.observe(2.,[],1.5)
        self.assertAlmostEqual(result['pruned_mass'],1-np.exp(full.log_weights).max(),places=12)


if __name__=='__main__':unittest.main()
