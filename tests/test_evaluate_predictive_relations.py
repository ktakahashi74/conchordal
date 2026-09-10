import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from evaluate_context_expectation import ContextExpectation
from evaluate_predictive_relations import PredictiveRelations, VARIANTS


CENTERS = np.log2([110.,220.,440.,880.,1760.])


def model(pairs=((0,2),(1,3))):
    return PredictiveRelations(CENTERS,pairs,[1],.05,.2,8,2.,.5,[.5,.5])


class PredictiveRelationsTests(unittest.TestCase):
    def test_own_predictions_match_existing_independent_model(self):
        relation = model()
        independent = ContextExpectation(5,[1],.05,.2,8,[0.,0.],np.eye(2),2.,.5,[.5,.5])
        for i,value in enumerate(np.random.default_rng(731).uniform(0.,1.,size=(40,5))):
            if i % 9 == 0:
                value[0] = 0.
            result = relation.observe((i+1)*.05,value)
            expected = independent.observe((i+1)*.05,value)
            if result["kind"] == "score":
                np.testing.assert_array_equal(result["log_density"]["own"],expected["log_density"][relation.pairs])
                self.assertEqual(result["conditional_partner_gain_bits"].shape,(2,2))

    def test_background_excludes_both_band_neighborhoods(self):
        relation = model(pairs=((0,2),))
        self.assertEqual(relation.background_weights[0,0],0.)
        self.assertEqual(relation.background_weights[0,2],0.)
        np.testing.assert_allclose(relation.background_weights.sum(axis=1),1.)
        relation.observe(.05,np.log1p([1.,2.,3.,4.,5.]))
        gathered = relation.models["both"].recent[-1]
        expected = np.log1p(np.sqrt((2**2+4**2+5**2)/3))
        self.assertAlmostEqual(gathered[2],expected)

    def test_partner_is_lagged_and_does_not_enter_own_prediction(self):
        a,b = model(pairs=((0,2),)),model(pairs=((0,2),))
        for i in range(30):
            value = np.array([.1+(i%3)*.1,.2,.2+(i%4)*.1,.3,.4])
            a.observe((i+1)*.05,value)
            b.observe((i+1)*.05,value)
        frozen = copy.deepcopy(a.forecast())
        changed = np.array([.2,.2,10.,.3,.4])
        a.observe(1.55,[.2,.2,.3,.3,.4])
        b.observe(1.55,changed)
        # First target's old forecast and own-only next forecast cannot read a new partner.
        for key in ["location","scale2","log_weights"]:
            np.testing.assert_array_equal(a.forecast()["own"][key][0],b.forecast()["own"][key][0])
        self.assertTrue(any(not np.array_equal(a.forecast()["both"][key][0],b.forecast()["both"][key][0])
                            for key in ["location","scale2"]))
        self.assertEqual(frozen["both"]["issued_sec"],1.5)

    def test_gap_and_validation_leave_all_models_consistent(self):
        relation = model()
        relation.observe(.05,[.1,.2,.3,.4,.5])
        before = copy.deepcopy(relation.forecast())
        for bad in [[.1],[-1.,0.,0.,0.,0.],[np.nan,0.,0.,0.,0.]]:
            with self.assertRaises(ValueError):
                relation.observe(.1,bad)
        for name in VARIANTS:
            for key in before[name]:
                np.testing.assert_array_equal(relation.forecast()[name][key],before[name][key])
        result = relation.observe(2.,None)
        self.assertEqual(result["kind"],"input_gap")
        self.assertIsNone(relation.forecast())
        self.assertEqual(relation.observe(2.05,[.1,.2,.3,.4,.5])["kind"],"warmup")
        self.assertEqual(relation.observe(2.1,[.1,.2,.3,.4,.5])["kind"],"score")


if __name__ == "__main__":
    unittest.main()
