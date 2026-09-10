import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_relational_contexts import run
from evaluate_spectral_relations import run as native_run


class RelationalContextStreamTests(unittest.TestCase):
    def test_delayed_stream_scores_issued_mixtures_and_preserves_pending_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'source.gz', Path(directory)/'output.gz'
            rows = [dict(kind='contract',sample_rate=10,pairs=[[0,1],[1,0]])]
            queries = {}
            completed = 0
            for i in range(1,11):
                if i>3:
                    old = queries[i-3]
                    rows.append(dict(kind='completed',query_id=i-3,target_sample=i*10,
                                     features=old['features'],observed=[1+.1*i,.5+.03*i]))
                    completed += 1
                features = dict(background=[[math.sin(i)],[math.cos(i)]],
                                both=[[math.sin(i),math.cos(i)],[math.cos(i),math.sin(i)]])
                row = dict(kind='issued',query_id=i,issued_sample=i*10,target_sample=(i+3)*10,
                           features=features,completed_before_issue=[completed]*2)
                queries[i]=row;rows.append(row)
            rows.append(dict(kind='eof',pending=3))
            with gzip.open(source,'wt') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            run(source,output,max_paths=8)
            with gzip.open(output,'rt') as f:result=[json.loads(line) for line in f]
            issued={r['query_id']:r for r in result if r['kind']=='issued'}
            mismatches=0
            for row in result:
                if row['kind']!='completed':continue
                query=issued[row['query_id']]
                for name,forecasts in query['forecasts'].items():
                    for pair,(forecast,y) in enumerate(zip(forecasts,row['observed'],strict=True)):
                        total=0.
                        for mean,scale2,degree,weight,positive in zip(
                            forecast['location'],forecast['scale2'],forecast['degrees_freedom'],
                            forecast['log_weights'],forecast['positive_probability'],strict=True):
                            logp=(math.lgamma((degree+1)/2)-math.lgamma(degree/2)
                                  -.5*math.log(degree*math.pi*scale2)
                                  -(degree+1)/2*math.log1p((math.log(y)-mean)**2/(degree*scale2)))
                            total+=math.exp(weight+logp)*positive/y
                        self.assertAlmostEqual(row['issued_log_density'][name][pair],math.log(total),places=11)
                        mismatches+=abs(math.log(total)-row['filter_updates'][name][pair]['filter_log_density'])>1e-6
            self.assertGreater(mismatches,10)
            self.assertEqual(result[-1],dict(kind='eof',completed=7,pending=3,pending_targets=[110,120,130]))

    def test_opt_in_native_features_include_eof_queries_without_changing_old_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'native.jsonl'
            frequencies=np.arange(100.,3100.,100.)
            power=np.zeros(len(frequencies));power[[1,10,24]]=[3.,2.,1.]
            with source.open('w') as f:
                f.write(json.dumps(dict(sample_rate=100,hop_samples=10,window_samples=10,
                                        frequency_hz=frequencies.tolist()))+'\n')
                for i in range(1,11):
                    current=power*(1+.01*i)
                    f.write(json.dumps(dict(available_sec=i*.1,full_window=True,nsgt_power_scan=current.tolist(),
                                            known_rms_by_age_scan=np.tile(np.sqrt(current)[:,None],(1,8)).tolist(),
                                            history_observed_through_sample=i*10,known_coverage_by_age=[1.]*8))+'\n')
            native_run(source,root/'old.gz',prefix_sec=.2,horizon_sec=.2)
            native_run(source,root/'new.gz',prefix_sec=.2,horizon_sec=.2,include_issued_features=True)
            with gzip.open(root/'old.gz','rt') as a,gzip.open(root/'new.gz','rt') as b:
                old,new=[json.loads(line) for line in a],[json.loads(line) for line in b]
            issued=[r for r in new if r['kind']=='issued'];done={r['query_id'] for r in new if r['kind']=='completed'}
            pending=[r for r in issued if r['query_id'] not in done]
            self.assertEqual(len(pending),new[-1]['pending'])
            self.assertGreater(len(pending),0)
            self.assertTrue(all(len(r['features']['both'])==6 for r in pending))
            for row in new:
                if row['kind']=='issued':row.pop('features')
            self.assertEqual(old,new)


if __name__=='__main__':unittest.main()
