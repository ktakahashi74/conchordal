"""Freeze addendum-5 classifications and direct-PCM controls; never edit originals."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.input/'inputs-manifest.json').read_text())
erb=lambda f:9.2645*math.log1p(.00437*f)
records=[]
for fixture in manifest['fixtures']:
    f=dict(fixture)
    pairs=[]
    for i,j in itertools.combinations(range(len(f['frequencies'])),2):
        e1,e2=map(erb,(f['frequencies'][i],f['frequencies'][j]))
        difference=e2-e1
        pairs.append(dict(components=[i,j],difference_erb=difference,
            resolution='resolved' if difference>=.75 else 'unresolved'))
    f['pairs']=pairs
    f['pair_input_resolution']=pairs[0]['resolution'] if len(f['frequencies'])==2 else ('single' if not pairs else 'mixed_complex')
    src=a.input/'reference'/f['pcm']
    data=src.read_bytes()
    assert hashlib.sha256(data).hexdigest()==f['pcm_sha256']
    assert len(data)==4*f['length_samples']
    assert f['length_samples']//512>=f['onset_hop']+47
    shutil.copy2(src,a.out/f['pcm'])
    f['judgment_hops']=f['observation_hops']
    first=f['onset_hop'] if f['category']=='transient' else f['length_samples']//512-47
    f['flora_hops']=list(range(first,first+47))
    f['flora_hops_8']=f['observation_hops']
    records.append(f)
controls=[]
for f in records:
    if f['name'] not in ['loo_110.0_40.0_ratio1.0','loo_110.0_40.0_ratio0.5']:
        continue
    t=np.arange(f['length_samples'])/48000
    direct=sum(amp*np.cos(2*np.pi*freq*t+phase) for freq,amp,phase in zip(f['frequencies'],f['amplitudes'],f['phases'])).astype('<f4')
    own=(.11*np.cos(2*np.pi*440*t+.22)).astype(np.float32)
    residual=((direct+own).astype(np.float32)-own).astype('<f4')
    assert residual.tobytes()==(a.out/f['pcm']).read_bytes()
    name=f['name']+'_direct'
    out=dict(f,name=name,pcm=name+'.f32',pcm_sha256=hashlib.sha256(direct.tobytes()).hexdigest(),control_for=f['name'])
    direct.tofile(a.out/out['pcm'])
    out['residual_vs_direct_max_abs']=float(np.max(np.abs(residual-direct)))
    out['residual_vs_direct_relative_l2']=float(np.linalg.norm((residual-direct).astype(float))/np.linalg.norm(direct.astype(float)))
    controls.append(out)
manifest.update(fixtures=records,direct_controls=controls,classification='pairwise LTFAT ERB difference >=0.75, no tolerance',
    flora_window_hops=47,input_extension='none needed: existing 64-hop inputs have >=47 hops after onset; original bytes and SHA256 retained')
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'fixtures':len(records),'direct_controls':len(controls),'loo_110_150':[f['pairs'] for f in records if f['name'].startswith('loo_110.0_40.0')],'direct_differences':[f['residual_vs_direct_max_abs'] for f in controls]}))
