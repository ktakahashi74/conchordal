"""Joint continuation and context corrections conditional on the currently heard level."""

import math

import numpy as np
from scipy.special import logsumexp

from context_paths import branches,transition


class ContinuationContexts:
    def __init__(self,target_dimensions,relation_dimensions,*,mode,shared_continuation,
                 correction_sd,residual_sd,observation_sd,max_paths,contexts=3,change_rate_hz=.2):
        if (any(not isinstance(d,int) or d<1 for d in (target_dimensions,relation_dimensions))
                or mode not in ('stationary','renewal','recurrent')
                or not isinstance(shared_continuation,bool)
                or not isinstance(contexts,int) or contexts<2
                or not isinstance(max_paths,int) or max_paths<1
                or not math.isfinite(change_rate_hz) or change_rate_hz<0
                or not math.isfinite(correction_sd) or correction_sd<=0
                or any(not math.isfinite(v) or v<0 for v in (residual_sd,observation_sd))
                or residual_sd**2+observation_sd**2<=0):
            raise ValueError('expected explicit dimensions, priors, noise, contexts and numerical budget')
        self.target_dimensions,self.relation_dimensions=target_dimensions,relation_dimensions
        self.mode,self.shared_continuation=mode,shared_continuation
        self.contexts=contexts if mode=='recurrent' else 1
        self.rate,self.max_paths=change_rate_hz,max_paths
        self.residual_variance,self.observation_variance=residual_sd**2,observation_sd**2
        self.coefficient_variance=correction_sd**2/2
        self.shared_size=target_dimensions if shared_continuation else 0
        self.local_size=relation_dimensions if shared_continuation else target_dimensions+relation_dimensions
        size=self.shared_size+self.contexts*self.local_size
        self.mean=np.zeros((1,size))
        self.cov=np.eye(size)[None,:,:]*self.coefficient_variance
        self.active=np.zeros(1,dtype=int)
        self.used=np.zeros(1,dtype=int)
        self.log_weights=np.zeros(1)
        self.clock_sec=float('-inf')
        self.last_target_sec=None
        self.completed=0

    def transition(self,elapsed_sec):
        return transition(self.contexts,self.rate,elapsed_sec)

    def _components(self,target_sec,target_features,relation_features,offset):
        own=np.asarray(target_features,dtype=float)
        relation=np.asarray(relation_features,dtype=float)
        if (not math.isfinite(target_sec) or target_sec<=self.clock_sec
                or not math.isfinite(offset) or offset<0
                or own.shape!=(self.target_dimensions,) or relation.shape!=(self.relation_dimensions,)
                or not np.isfinite(own).all() or not np.isfinite(relation).all()):
            raise ValueError('expected future target, finite frozen features and nonnegative observed offset')
        elapsed=0. if self.last_target_sec is None else target_sec-self.last_target_sec
        route=branches(self.mode,self.contexts,self.rate,self.active,self.used,
                       elapsed,self.last_target_sec is None)
        parent,state=route['parent'],route['state']
        mean,cov=self.mean[parent].copy(),self.cov[parent].copy()
        design=np.zeros_like(mean)
        if self.shared_size:design[:,:self.shared_size]=own
        local=relation if self.shared_size else np.r_[own,relation]
        for i,context in enumerate(state):
            first=self.shared_size+context*self.local_size
            last=first+self.local_size
            design[i,first:last]=local
            if route['reset'][i]:
                # Replace one block independently, preserving the retained marginal.
                mean[i,first:last]=0.
                cov[i,first:last,:]=0.
                cov[i,:,first:last]=0.
                cov[i,first:last,first:last]=np.eye(self.local_size)*self.coefficient_variance
        cov_x=np.einsum('nij,nj->ni',cov,design)
        parameter_variance=np.einsum('ni,ni->n',design,cov_x)
        return dict(route,mean=mean,cov=cov,design=design,cov_x=cov_x,
                    location=offset+np.einsum('ni,ni->n',mean,design),
                    parameter_variance=parameter_variance,
                    variance=parameter_variance+self.residual_variance+self.observation_variance,
                    log_weights=self.log_weights[parent]+route['log_transition'])

    def forecast(self,target_sec,target_features,relation_features,offset):
        c=self._components(target_sec,target_features,relation_features,offset)
        return dict(target_sec=target_sec,completed=self.completed,observed_offset=offset,
                    location=c['location'],log_weights=c['log_weights'],
                    within_component_parameter_variance=c['parameter_variance'],
                    residual_variance=self.residual_variance,
                    observation_noise_variance=self.observation_variance,
                    component_observation_variance=c['variance'])

    def observe(self,target_sec,target_features,relation_features,offset,value):
        if not math.isfinite(value) or value<0:
            raise ValueError('expected a finite nonnegative actual observation')
        c=self._components(target_sec,target_features,relation_features,offset)
        signs=np.array([value,-value]) if value>0 else np.array([0.])
        log_normal=-.5*((signs[:,None]-c['location'])**2/c['variance']+np.log(2*math.pi*c['variance']))
        if value==0:log_normal+=math.log(2.)
        joint=(log_normal+c['log_weights']).reshape(-1)
        evidence=float(logsumexp(joint));posterior=joint-evidence
        index=np.tile(np.arange(len(c['parent'])),len(signs))
        error=np.repeat(signs,len(c['parent']))-c['location'][index]
        mean=c['mean'][index]+c['cov_x'][index]*(error/c['variance'][index])[:,None]
        updated_cov=c['cov']-c['cov_x'][:,:,None]*c['cov_x'][:,None,:]/c['variance'][:,None,None]
        cov=updated_cov[index].copy()
        state=c['state'][index]
        used=np.maximum(self.used[c['parent'][index]],state+1)
        mean[mean==0]=0.
        groups={};representatives=[];merged=[]
        for i in range(len(index)):
            key=(int(state[i]),int(used[i]),mean[i].tobytes(),cov[i].tobytes())
            if key in groups:
                group=groups[key];merged[group]=np.logaddexp(merged[group],posterior[i])
            else:
                groups[key]=len(merged);representatives.append(i);merged.append(posterior[i])
        merged=np.array(merged)
        keep=np.argsort(-merged,kind='stable')[:self.max_paths]
        retained=float(logsumexp(merged[keep]))
        selected=np.array(representatives)[keep]
        self.mean,self.cov=mean[selected].copy(),cov[selected].copy()
        self.active,self.used=state[selected].copy(),used[selected].copy()
        self.log_weights=merged[keep]-retained
        self.clock_sec=self.last_target_sec=target_sec
        self.completed+=1
        return dict(filter_log_density=evidence,pruned_mass=max(0.,-math.expm1(retained)),
                    retained_paths=len(keep),merged_paths=len(merged),
                    new_context_mass=float(np.exp(posterior[c['fresh'][index]]).sum()),
                    returning_context_mass=float(np.exp(posterior[c['returning'][index]]).sum()))

    def skip_to(self,end_sec):
        if not math.isfinite(end_sec) or end_sec<=self.clock_sec:
            raise ValueError('expected an advancing observation clock')
        self.clock_sec=end_sec
