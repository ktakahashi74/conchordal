import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.integrate import quad

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uncertain_regression import bounded_crps,cdf
from evaluate_relational_contexts import run
from evaluate_context_scores import run as score_run


class UncertainScoreTests(unittest.TestCase):
    def test_partitioned_score_matches_independent_density_integral(self):
        for locations,variances,weights in [([.5,1.4],[.01,.3],[.7,.3]),
                                            ([.7],[1e-9],[1.]),([0.],[50.],[1.])]:
            f=dict(location=np.array(locations),component_observation_variance=np.array(variances),
                   log_weights=np.log(weights))
            def probability(u):
                y=u/(1-u) if u<1 else math.inf
                value=0.
                for mu,var,w in zip(locations,variances,weights,strict=True):
                    value+=w*.5*(math.erf((y-mu)/math.sqrt(2*var))-math.erf((-y-mu)/math.sqrt(2*var)))
                return value
            for y in [0.,1e-12,.6,1.1,20.]:
                at=y/(1+y)
                anchors=[max(0.,abs(mu)+j*math.sqrt(v)) for mu,v in zip(locations,variances) for j in [-8,-2,0,2,8]]
                edges=sorted(set([0.,at,1.,*[a/(1+a) for a in anchors]]))
                reference=sum(quad(lambda u:(probability(u)-(1. if lo>=at else 0.))**2,
                                   lo,hi,epsabs=1e-12)[0] for lo,hi in zip(edges[:-1],edges[1:]))
                self.assertAlmostEqual(bounded_crps(f,y,order=32),reference,delta=3e-7)
            np.testing.assert_allclose(cdf(f,[0.,.2,1.,10.])+cdf(f,[0.,.2,1.,10.],survival=True),1.,atol=1e-14)

    def test_score_units_component_splitting_and_serialized_forecasts(self):
        original=dict(location=[.2,1.1],component_observation_variance=[.03,.2],log_weights=np.log([.3,.7]).tolist())
        split=dict(location=[.2,1.1,1.1],component_observation_variance=[.03,.2,.2],log_weights=np.log([.3,.2,.5]).tolist())
        scaled=dict(location=np.array(original['location'])*12,
                    component_observation_variance=np.array(original['component_observation_variance'])*144,
                    log_weights=original['log_weights'])
        self.assertAlmostEqual(bounded_crps(original,.7),bounded_crps(split,.7),places=13)
        self.assertAlmostEqual(bounded_crps(original,.7),bounded_crps(scaled,8.4,scale=12),places=13)
        with self.assertRaises(ValueError):bounded_crps(original,-1.)

    def test_stream_freezes_noise_predictions_and_matches_zero_missing_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,output,scores=[Path(tmp)/n for n in ('source.gz','predictions.gz','scores.gz')]
            features=dict(background=[[.2]*9],both=[[.2]*18])
            rows=[dict(kind='contract',sample_rate=10,pairs=[[0,1]])]
            queries={}
            for i in range(1,7):
                if i>2:
                    q=queries[i-2]
                    rows.append(dict(kind='completed',query_id=i-2,target_sample=i*10,
                                     features=features,observed=[0. if i%2 else 1.]))
                q=dict(kind='issued',query_id=i,issued_sample=i*10,target_sample=(i+2)*10,
                       completed_before_issue=[max(0,i-2)],features=features)
                queries[i]=q;rows.append(q)
            rows.append(dict(kind='censored',query_id=5,reason='missing_input'))
            rows.append(dict(kind='eof',pending=1))
            with gzip.open(source,'wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            saved=source.read_bytes()
            run(source,output,max_paths=16,observation_sd=.2)
            score_run(output,scores,scales=(1.,))
            with gzip.open(output,'rt') as f:predictions=[json.loads(line) for line in f]
            with gzip.open(scores,'rt') as f:losses=[json.loads(line) for line in f]
            issued={r['query_id']:r for r in predictions if r['kind']=='issued'}
            mismatch=0
            for row in predictions:
                if row['kind']!='completed':continue
                q=issued[row['query_id']]
                for name,forecasts in q['forecasts'].items():
                    f=forecasts[0];y=row['observed'][0]
                    density=sum(math.exp(w)/math.sqrt(2*math.pi*v)*
                                (math.exp(-.5*(y-m)**2/v)+math.exp(-.5*(y+m)**2/v))
                                for m,v,w in zip(f['location'],f['component_observation_variance'],f['log_weights'],strict=True))
                    self.assertAlmostEqual(row['issued_log_density'][name][0],math.log(density),places=11)
                    mismatch+=abs(math.log(density)-row['filter_updates'][name][0]['filter_log_density'])>1e-7
                score=next(r for r in losses if r['kind']=='completed' and r['query_id']==row['query_id'])
                self.assertEqual(score['observed'],row['observed'])
                self.assertAlmostEqual(score['persistence'][0],.6,places=14)
            self.assertGreater(mismatch,0)
            self.assertEqual(predictions[-1]['pending_targets'],[80])
            self.assertEqual(losses[-1]['censored'],1)
            self.assertEqual(losses[-1]['completed'],4)
            self.assertEqual(source.read_bytes(),saved)


if __name__=='__main__':unittest.main()
