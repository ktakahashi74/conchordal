import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_spectral_relations import SpectralRelations, VARIANTS, regions


def model():
    return SpectralRelations(np.eye(4)[:3], 1., 100, 10, ridge=1.)


def observation(i):
    x = np.array([.4 + .2*np.sin(i*.27), .3 + .2*np.cos(i*.15),
                  .2 + .1*np.sin(i*.73), .1])
    history = np.array([x*(.7 + .2*np.sin((i-age)*.27)) for age in range(8)]).T
    return x*x, history, np.linspace(.95, .05, 8)


class NativeRelationTests(unittest.TestCase):
    def test_completed_only_fit_matches_independent_batch_solution(self):
        m = model()
        training = [[] for _ in m.pairs]
        observed = [[] for _ in m.pairs]
        compared = 0
        for i in range(110):
            for event in m.observe((i+1)*10, *observation(i)):
                if event["kind"] == "completed":
                    for pair in range(len(m.pairs)):
                        training[pair].append({n: event["features"][n][pair] for n in VARIANTS})
                        observed[pair].append(event["observed"][pair])
                elif event["kind"] == "issued" and event["completed_before_issue"][0]:
                    query = m.pending[-1]
                    for pair in range(len(m.pairs)):
                        y = np.array(observed[pair]);self.assertEqual(len(y),event["completed_before_issue"][pair])
                        for name in VARIANTS:
                            x = np.array([r[name] for r in training[pair]])
                            xc = x-x.mean(axis=0)
                            coef = np.linalg.solve(xc.T@xc+np.eye(x.shape[1]),xc.T@(y-y.mean()))
                            expected=max(0., y.mean()+(query["features"][name][pair]-x.mean(axis=0))@coef)
                            self.assertAlmostEqual(expected,event["predictions"][name][pair],places=11)
                            compared+=1
        self.assertGreater(compared,2000)

    def test_issue_is_immutable_and_future_inputs_cannot_change_common_prefix(self):
        a,b=model(),model()
        for i in range(25):
            self.assertEqual(a.observe((i+1)*10,*observation(i)),b.observe((i+1)*10,*observation(i)))
        issued = a.observe(260,*observation(25))[-1]
        b.observe(260,*observation(25))
        expected=json.loads(json.dumps(issued))
        issued["predictions"]["own"][0]=999
        for i in range(26,36):
            events=a.observe((i+1)*10,*observation(i+100))
        completed=next(e for e in events if e["kind"]=="completed" and e["query_id"]==expected["query_id"])
        self.assertEqual(completed["predictions"],expected["predictions"])
        self.assertNotEqual(a.models[0]["own"].mean_y.tolist(),b.models[0]["own"].mean_y.tolist())

    def test_gap_censors_pending_without_training_or_erasing_previous_learning(self):
        m=model()
        for i in range(22):m.observe((i+1)*10,*observation(i))
        n=m.models[0]["own"].n;pending=len(m.pending)
        events=m.observe(300,None)
        self.assertEqual(len(events),pending)
        self.assertTrue(all(e["kind"]=="censored" for e in events))
        self.assertEqual(m.models[0]["own"].n,n)
        self.assertEqual(len(m.pending),0)
        m.observe(310,*observation(30))
        self.assertEqual(m.models[0]["own"].n,n)
        self.assertEqual(m.pending[0]["target_sample"],410)

    def test_silence_and_unknown_history_remain_distinct(self):
        m=model()
        m.observe(10,*observation(0))
        for end in range(20,111,10):
            events=m.observe(end,np.zeros(4))
        completed=next(e for e in events if e["kind"]=="completed")
        self.assertEqual(completed["observed"],[0.]*len(m.pairs))
        self.assertEqual(m.models[0]["own"].n,1)
        self.assertEqual(len(m.pending),0)

    def test_partner_supplies_delayed_information_beyond_own_and_background(self):
        rng=np.random.default_rng(3)
        source=rng.uniform(.05,2.,900)
        independent=rng.uniform(.05,2.,900)
        background=rng.uniform(.05,2.,900)
        m=model();sse={name:0. for name in [*VARIANTS,"mean","persistence"]}
        count=0
        for i in range(len(source)):
            rms=np.array([source[i],source[max(0,i-10)],background[i],independent[i]])
            h=np.tile(rms[:,None],(1,8))
            for e in m.observe((i+1)*10,rms*rms,h,np.ones(8)):
                if e["kind"]=="completed" and i>450:
                    pair=m.pairs.index((1,0))
                    for name in sse:sse[name]+=(e["predictions"][name][pair]-e["observed"][pair])**2
                    count+=1
        self.assertGreater(count,400)
        self.assertLess(sse["both"],.02*sse["background"])
        self.assertLess(sse["partner"],.02*sse["own"])

    def test_regions_are_observed_peaks_without_known_source_labels(self):
        frequencies=np.arange(100.,3100.,100.)
        power=np.zeros(len(frequencies));power[[1,10,24]]=[3,2,1]
        indices,weights=regions(frequencies,power,3)
        self.assertEqual(indices,[1,10,24])
        np.testing.assert_allclose(weights.sum(axis=1),1)
        self.assertIsNone(regions(frequencies,np.zeros(len(frequencies)),3))

    def test_older_partner_information_is_distinct_from_repeated_current_values(self):
        rng=np.random.default_rng(31)
        source=rng.uniform(.05,2.,900)
        rms=np.column_stack([source,np.r_[np.full(20,source[0]),source[:-20]],
                             rng.uniform(.05,2.,900),rng.uniform(.05,2.,900)])
        m=model();sse={name:0. for name in ("both","both_current")}
        for i in range(len(source)):
            h=np.array([rms[max(0,i-lag)] for lag in (0,2,4,6,8,10,15,20)]).T
            for e in m.observe((i+1)*10,rms[i]**2,h,np.ones(8)):
                if e["kind"]=="completed" and i>450:
                    pair=m.pairs.index((1,0))
                    for name in sse:sse[name]+=(e["predictions"][name][pair]-e["observed"][pair])**2
        self.assertLess(sse["both"],.1*sse["both_current"])

    def test_rejects_misaligned_native_history(self):
        with self.assertRaisesRegex(ValueError,"native history"):
            model().observe(10,np.ones(4),np.ones((3,8)),np.ones(8))
