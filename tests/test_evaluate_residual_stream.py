import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_residual_uncertainty import run
from evaluate_context_scores import run as score_run


class ResidualStreamTests(unittest.TestCase):
    def test_scores_use_issued_distributions_and_missing_input_does_not_update(self):
        rows = [dict(kind='contract', sample_rate=10, pairs=[[0, 1]])]
        queries = {}
        for i in range(1, 7):
            if i == 4:
                rows.append(dict(kind='censored', query_id=2))
            if i > 2 and i != 4:
                old = queries[i-2]
                rows.append(dict(kind='completed', query_id=i-2, target_sample=i*10,
                                 features=old['features'], observed=[0. if i == 5 else .2*i]))
            own = [.2*i]+[.1*j for j in range(1, 17)]
            outside, partner = [.03*j for j in range(9)], [.02*j for j in range(9)]
            features = dict(background=[own+outside], both=[own+outside+partner])
            completed = sum(r['kind'] == 'completed' for r in rows)
            query = dict(kind='issued', query_id=i, issued_sample=i*10, target_sample=(i+2)*10,
                         features=features, completed_before_issue=[completed])
            rows.append(query)
            queries[i] = query
        rows.append(dict(kind='eof', pending=2))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, scores = (root/f'{s}.gz' for s in ('source', 'predictions', 'scores'))
            with gzip.open(source, 'wt') as f:
                for row in rows: f.write(json.dumps(row)+'\n')
            for drift in (None, 0., 1.):
                output, scores = (root/f'{label}-{drift}.gz' for label in ('predictions', 'scores'))
                run(source, output, state_drift_per_sec=drift)
                score_run(output, scores)
                with gzip.open(output, 'rt') as f: result = [json.loads(line) for line in f]
                with gzip.open(scores, 'rt') as f: evaluated = [json.loads(line) for line in f]
                issued = {r['query_id']: r for r in result if r['kind'] == 'issued'}
                self.assertEqual(result[0]['observation_model'], 'folded_student')
                self.assertEqual(result[-1], dict(kind='eof', completed=3, pending=2, pending_targets=[70, 80]))
                self.assertEqual(evaluated[-1], dict(kind='eof', issued=6, completed=3, censored=1,
                                                    pending=2, pending_targets=[70, 80]))
                for i, query in issued.items():
                    self.assertEqual(query['features'], queries[i]['features'])
                    for family, size in (('background', 28), ('both', 37), ('continuation', 0)):
                        x = query['conditioning'][family][0]
                        self.assertEqual(len(x['design']), size)
                        self.assertAlmostEqual(np.linalg.norm(x['design'])**2, 2. if size else 0.)
                        forecast = query['forecasts'][family][0]
                        self.assertEqual(forecast['completed'], queries[i]['completed_before_issue'][0])
                        self.assertEqual(forecast['degrees_freedom'], 6+forecast['completed'])
                    self.assertEqual(query['forecasts']['unlearned_continuation'][0]['completed'], 0)
                different = 0
                for row in result:
                    if row['kind'] != 'completed': continue
                    y = row['observed'][0]
                    for name, forecasts in issued[row['query_id']]['forecasts'].items():
                        f = forecasts[0]
                        terms = [t.logpdf(sign*y, f['degrees_freedom'], loc=f['location'],
                                          scale=np.sqrt(f['scale2']))+f['log_weights'] for sign in (-1, 1)]
                        expected = float(logsumexp(terms))
                        self.assertAlmostEqual(expected, row['issued_log_density'][name][0], places=11)
                        if name in row['filter_updates']:
                            different += abs(expected-row['filter_updates'][name][0]['filter_log_density']) > 1e-6
                self.assertGreater(different, 0)
                for row in evaluated:
                    if row['kind'] != 'completed': continue
                    for scored in row['scores']:
                        self.assertEqual(len(scored['loss']), 5)
                        scale = scored['scale']
                        current, actual = row['persistence'][0], row['observed'][0]
                        self.assertAlmostEqual(scored['loss']['persistence'][0],
                                               abs(current/(current+scale)-actual/(actual+scale)))


if __name__ == '__main__': unittest.main()
