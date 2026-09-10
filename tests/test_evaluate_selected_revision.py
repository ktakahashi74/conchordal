from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_perceptual_action import FIELDS
from evaluate_selected_revision import main
from evaluate_structural_predictions import build_driven_bank


class SelectedRevisionTests(unittest.TestCase):
    def test_common_comparison_and_future_cannot_enter_selected_proposals(self):
        original_read = np.fromfile
        def native_stub(config,cases,directory,label):
            # A bounded history-dependent stand-in tests access order, not native kernels.
            for case in cases:
                past = original_read(case['past'],dtype='<f4').astype(float)
                with Path(case['output']).open('x') as output:
                    output.write(json.dumps(dict(type='contract',id=case['id'],issued_sample=len(past),hop_samples=128))+'\n')
                    for branch in case['branches']:
                        future = original_read(branch['audio'],dtype='<f4').astype(float)
                        end = (len(past)+len(future))//128*128
                        observed = np.r_[past,future][end-128:end]
                        values = [.5+.2*np.tanh(10*observed.mean()),.5+.2*np.tanh(100*np.mean(observed**2))]
                        output.write(json.dumps(dict(id=branch['id'],observation=dict(end_sample=end,**{f:values for f in FIELDS})))+'\n')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'native/case';source.mkdir(parents=True)
            previous = root/'previous/group-0/case';previous.mkdir(parents=True)
            fs,issue,adapt,compare,horizon = 24000,1024,96,48,480
            total = adapt+compare+horizon
            t = np.arange(issue+total)/fs
            environment = .05*np.cos(2*np.pi*480*t)
            own_past = .015*np.sin(2*np.pi*1200*t[:issue])
            own_sound = .04*np.sin(2*np.pi*900*t[issue:])
            for name,value in dict(past_mix=environment[:issue]+own_past,past_own=own_past,
                                   own_wait=np.zeros(total),own_sound=own_sound).items():
                np.asarray(value,dtype='<f4').tofile(source/(name+'.f32le'))
            settings = dict(drive_per_sec=[0.,1000.],initial_precision=.001,variance_prior=[2.,1e-6])
            old_plan = dict(hypotheses=dict(adaptation_sec=adapt/fs,selection_sec=compare/fs,horizon_sec=total/fs),driven_body=settings)
            (root/'previous/plan.json').write_text(json.dumps(old_plan))
            start = issue-adapt-compare
            fit = SimpleNamespace(frequency_hz=np.array([480.]),log_gain_per_sec=np.array([-4.]),fit_end_sample=issue-compare)
            past = original_read(source/'past_mix.f32le',dtype='<f4').astype(float)-original_read(source/'past_own.f32le',dtype='<f4').astype(float)
            bank,rows = build_driven_bank([('refitted_passive',SimpleNamespace(trajectory=fit),dict(adaptation_start=start))],
                past[start:issue-compare],fs,start,settings)
            bank.observe(issue-compare,past[issue-compare:])
            (previous/'issued.json').write_text(json.dumps(dict(body_candidates=rows,forecast_sha256={'body_mixture':bank.forecast(total).digest()})))
            (root/'native/inputs.json').write_text(json.dumps([dict(name='case',fs=fs,issue_sample=issue,seed=185317)]))
            plan = dict(chosen_actions=['sound'],adaptation_sec=adapt/fs,comparison_sec=compare/fs,horizon_sec=horizon/fs,
                fresh_counts=[0,1],fresh_offsets_sec=[0.],driven_body=settings,revision_prior_mass=dict(retained=1.,fresh=1.),
                perceptual_sampling=dict(seed=185319,target_draws=8,min_per_component=2,max_omitted_mass=1e-8,alpha=.05))
            (root/'plan.json').write_text(json.dumps(plan))
            outputs=[]
            for attempt in range(5):
                sound = environment[issue:]+own_sound
                wait = environment[issue:].copy()
                if attempt==1:
                    sound[adapt+compare:] += .025
                if attempt==2:
                    sound[adapt:adapt+compare] += .03
                if attempt==3:
                    wait += .045
                if attempt==4:
                    sound[:adapt] += .04
                np.asarray(sound,dtype='<f4').tofile(source/'truth_sound.f32le')
                np.asarray(wait,dtype='<f4').tofile(source/'truth_wait.f32le')
                out = root/f'run-{attempt}'
                def guarded_read(path,*args,**kwargs):
                    if Path(path).name.startswith('truth_'):
                        self.assertEqual(Path(path).name,'truth_sound.f32le')
                        directory=out/'case.sound'
                        count=kwargs['count']
                        if count==adapt:
                            self.assertFalse((directory/'proposed.json').exists())
                        elif count==compare:
                            self.assertTrue((directory/'proposed.json').exists())
                            self.assertFalse((directory/'issued.json').exists())
                        else:
                            self.assertEqual(count,horizon)
                            self.assertTrue((directory/'prediction-seal.json').exists())
                    return original_read(path,*args,**kwargs)
                argv=['evaluate','--input',str(source.parent),'--previous',str(root/'previous'),
                      '--output',str(out),'--plan',str(root/'plan.json'),'--config',str(root/'config.toml')]
                with patch.object(sys,'argv',argv),patch('evaluate_selected_revision.native',native_stub), \
                        patch('evaluate_selected_revision.np.fromfile',guarded_read),redirect_stdout(io.StringIO()):
                    main()
                outputs.append(out/'case.sound')
            for index in (1,2,3):
                self.assertEqual((outputs[0]/'proposed.json').read_bytes(),(outputs[index]/'proposed.json').read_bytes())
            for index in (1,3):
                for name in ('issued.json','forecast-moments.npz','integration-moments.npz'):
                    self.assertEqual((outputs[0]/name).read_bytes(),(outputs[index]/name).read_bytes())
            self.assertNotEqual((outputs[0]/'issued.json').read_bytes(),(outputs[2]/'issued.json').read_bytes())
            self.assertNotEqual((outputs[0]/'proposed.json').read_bytes(),(outputs[4]/'proposed.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
