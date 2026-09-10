"""Folded observations with distinct coefficient, sign and context uncertainty."""

import math

import numpy as np
from scipy.special import logsumexp

from context_paths import branches, transition


class MagnitudeContexts:
    def __init__(self, dimensions, *, mode, residual_sd, observation_sd, max_paths,
                 contexts=3, change_rate_hz=.2, intercept_precision=.01, slope_precision=1.):
        if (not isinstance(dimensions,int) or dimensions<0
                or mode not in ('stationary','renewal','recurrent')
                or not isinstance(contexts,int) or contexts<2
                or not isinstance(max_paths,int) or max_paths<1
                or not math.isfinite(change_rate_hz) or change_rate_hz<0
                or any(not math.isfinite(v) or v<=0 for v in (intercept_precision,slope_precision))
                or any(not math.isfinite(v) or v<0 for v in (residual_sd,observation_sd))
                or residual_sd**2+observation_sd**2<=0):
            raise ValueError('expected explicit finite prior, noise, transition and budget parameters')
        self.dimensions,self.mode=dimensions,mode
        self.contexts=contexts if mode=='recurrent' else 1
        self.rate,self.max_paths=change_rate_hz,max_paths
        self.residual_variance,self.observation_variance=residual_sd**2,observation_sd**2
        self.prior_cov=np.diag(1/np.r_[intercept_precision,np.full(dimensions,slope_precision)])
        self.cov=np.broadcast_to(self.prior_cov,(1,self.contexts,dimensions+1,dimensions+1)).copy()
        self.information=np.zeros((1,self.contexts,dimensions+1))
        self.used=np.zeros(1,dtype=int)
        self.active=np.zeros(1,dtype=int)
        self.log_weights=np.zeros(1)
        self.clock_sec=float('-inf')
        self.last_target_sec=None
        self.completed=0

    def transition(self, elapsed_sec):
        return transition(self.contexts,self.rate,elapsed_sec)

    def _components(self,target_sec,design):
        x=np.asarray(design,dtype=float)
        if (not math.isfinite(target_sec) or target_sec<=self.clock_sec
                or x.shape!=(self.dimensions,) or not np.isfinite(x).all()):
            raise ValueError('expected a future target and finite frozen design')
        x=np.r_[1.,x]
        elapsed=0. if self.last_target_sec is None else target_sec-self.last_target_sec
        route=branches(self.mode,self.contexts,self.rate,self.active,self.used,
                       elapsed,self.last_target_sec is None)
        parent,state,reset=route['parent'],route['state'],route['reset']
        cov,information=self.cov[parent,state].copy(),self.information[parent,state].copy()
        cov[reset]=self.prior_cov
        information[reset]=0.
        cov_x=cov@x
        parameter_variance=cov_x@x
        location=np.einsum('ij,ij->i',information,cov_x)
        return dict(x=x,parent=parent,state=state,cov=cov,information=information,cov_x=cov_x,
                    fresh=route['fresh'],returning=route['returning'],location=location,
                    parameter_variance=parameter_variance,
                    variance=parameter_variance+self.residual_variance+self.observation_variance,
                    log_weights=self.log_weights[parent]+route['log_transition'])

    def forecast(self,target_sec,design):
        c=self._components(target_sec,design)
        return dict(target_sec=target_sec,completed=self.completed,location=c['location'],
                    log_weights=c['log_weights'],
                    within_component_parameter_variance=c['parameter_variance'],
                    residual_variance=self.residual_variance,
                    observation_noise_variance=self.observation_variance,
                    component_observation_variance=c['variance'])

    def observe(self,target_sec,design,value):
        if not math.isfinite(value) or value<0:
            raise ValueError('expected finite nonnegative observation')
        c=self._components(target_sec,design)
        signs=np.array([value,-value]) if value>0 else np.array([0.])
        log_normal=-.5*((signs[:,None]-c['location'])**2/c['variance']
                       +np.log(2*math.pi*c['variance']))
        if value==0:
            log_normal+=math.log(2.)
        joint=(log_normal+c['log_weights']).reshape(-1)
        evidence=float(logsumexp(joint))
        posterior=joint-evidence
        index=np.tile(np.arange(len(c['parent'])),len(signs))
        parent,state=c['parent'][index],c['state'][index]
        cov,information=self.cov[parent].copy(),self.information[parent].copy()
        rows=np.arange(len(parent))
        cov[rows,state]=(c['cov']-c['cov_x'][:,:,None]*c['cov_x'][:,None,:]/c['variance'][:,None,None])[index]
        signed_values=np.repeat(signs,len(c['parent']))
        information[rows,state]=c['information'][index]+signed_values[:,None]*c['x']/(self.residual_variance+self.observation_variance)
        # Zero-mean priors make a whole coefficient vector's sign unidentifiable.
        pivot=np.argmax(abs(information),axis=-1)
        direction=np.take_along_axis(information,pivot[...,None],axis=-1)[...,0]
        information*=np.where(direction<0,-1.,1.)[...,None]
        information[information==0]=0.
        used=np.maximum(self.used[parent],state+1)
        groups={}
        representatives=[]
        merged=[]
        for i in range(len(parent)):
            key=(int(state[i]),int(used[i]),cov[i].tobytes(),information[i].tobytes())
            if key in groups:
                group=groups[key]
                merged[group]=np.logaddexp(merged[group],posterior[i])
            else:
                groups[key]=len(merged)
                representatives.append(i)
                merged.append(posterior[i])
        merged=np.array(merged)
        keep=np.argsort(-merged,kind='stable')[:self.max_paths]
        retained=float(logsumexp(merged[keep]))
        selected=np.array(representatives)[keep]
        self.cov,self.information=cov[selected].copy(),information[selected].copy()
        self.active,self.used=state[selected].copy(),used[selected].copy()
        self.log_weights=merged[keep]-retained
        self.clock_sec=self.last_target_sec=target_sec
        self.completed+=1
        return dict(filter_log_density=evidence,pruned_mass=max(0.,-math.expm1(retained)),
                    merged_paths=len(merged),retained_paths=len(keep),
                    new_context_mass=float(np.exp(posterior[c['fresh'][index]]).sum()),
                    returning_context_mass=float(np.exp(posterior[c['returning'][index]]).sum()))

    def skip_to(self,end_sec):
        if not math.isfinite(end_sec) or end_sec<=self.clock_sec:
            raise ValueError('expected an advancing observation clock')
        self.clock_sec=end_sec
