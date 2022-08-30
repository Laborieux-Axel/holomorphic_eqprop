import jax.numpy as jnp
import numpy as np
import haiku as hk
from haiku.data_structures import to_mutable_dict, to_immutable_dict
from jax import grad, jit, vmap, pmap, lax
from functools import partial
from jax.nn import log_softmax
from jax import random
from jax.lax import pmean, psum, top_k
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

import os

from utils.functions import dict_zeros_like, add_param_dict, nanify
from utils.functions import dict_none_like, dict_zero_like, mul_param_dict
from utils.functions import to_real_dict, div_param_dict, to_complex
from utils.functions import to_complex_dict, to_conjugate, to_conjugate_dict
from utils.functions import to_2real_dict, L2
from utils.data import printLog, split, split_batch, one_hot, par_one_hot
from utils.custom import SfmPool





class Identity(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        return x




class xent_phi(hk.Module):
    def __init__(self, archi, act, name=None):
        super().__init__(name=name)

        self.act = act
        self.convlen = archi['convlen']        
        self.fclen = archi['fclen']

        layers = []
        pools = []
        for i in range(self.convlen):

            l_name = 'xent_phi/~/conv_'+str(i+1)

            c = archi[l_name]['channel']
            k = archi[l_name]['kernel']
            s = archi[l_name]['stride']
            pd = archi[l_name]['padding']

            if archi[l_name].get('binit') is not None:
                b_ini = hk.initializers.Constant(archi[l_name]['binit'])
            else:
                b_ini = hk.initializers.Constant(0.0)

            if archi[l_name].get('sc') is not None:
                sc = archi[l_name]['sc']
                ini = hk.initializers.VarianceScaling(scale=sc, mode='fan_avg')
                layers.append(hk.Conv2D(c, k, stride=s, padding=pd, w_init=ini,
                                        b_init=b_ini, name='conv_'+str(i+1)))
            else:
                layers.append(hk.Conv2D(c, k, stride=s, padding=pd, b_init=b_ini,
                                        name='conv_'+str(i+1)))

            p = archi[l_name]['pooling']
            if p=='m':
                pools.append(hk.MaxPool((2,2,1), (2,2,1), padding='VALID'))
            elif p=='i':
                pools.append(Identity())
            elif p=='a':
                pools.append(hk.AvgPool((2,2,1), (2,2,1), padding='VALID'))
            elif p=='s':
                Tmp = archi[l_name]['Tmp']
                pools.append(SfmPool((2,2,1), (2,2,1), padding='VALID', T=Tmp))
                    
        for i in range(self.fclen): 
            l_name = 'xent_phi/~/fc_'+str(i+1+self.convlen)
            size = archi[l_name]['fc']

            if archi[l_name].get('binit') is not None:
                b_ini = hk.initializers.Constant(archi[l_name]['binit'])
            else:
                b_ini = hk.initializers.Constant(0.0)

            if archi[l_name].get('sc') is not None:
                sc = archi[l_name]['sc']
                ini = hk.initializers.VarianceScaling(scale=sc, mode='fan_avg')
                layers.append(hk.Linear(size, w_init=ini, b_init=b_ini,
                                        name='fc_'+str(i+1+self.convlen)))
            else:
                layers.append(hk.Linear(size, name='fc_'+str(i+1+self.convlen),
                                        b_init=b_ini))

        self.layers = tuple(layers)
        self.pools = tuple(pools)

    def __call__(self, x, h, y, beta):
        
        phi = 0.0
        l = x, *h

        for i in range(self.convlen):
            m = self.layers[i]
            p = self.pools[i]
            phi = phi + jnp.sum( p( m( l[i] )) * l[i+1])

        for i in range(self.convlen, len(self.layers)-1):
            m = self.layers[i]
            phi = phi + jnp.sum( m( l[i].flatten() ) * l[i+1] )

        # softmax readout
        logits = self.layers[-1]( l[-1].flatten() )
        xent = -y*log_softmax(logits)

        phi = phi - beta * jnp.sum(xent)

        return phi, logits


    def smart_init(self, x):

        out = []
        
        for i in range(self.convlen):
            m = self.layers[i]
            p = self.pools[i]
            x = self.act( p( m(x) ) )
            out.append(x)

        x = x.flatten()

        for i in range(self.convlen, len(self.layers)-1):
            m = self.layers[i]
            x = self.act( m(x) )
            out.append(x)

        return tuple(out)





class xent_p_cnn:

    def __init__(self, archi, act, n_targets, seed, n_dev):

        self.archi = archi
        self.act = act 
        self.n_targets = n_targets
        self.seed = seed
        self.n_devices = n_dev
        self.n_layers = archi['convlen'] + archi['fclen'] - 1

        def _phi(x, h, y, beta):
            module = xent_phi(archi, act)
            return module(x, h, y, beta)
        
        phi = hk.transform(_phi)
        self.phi = phi
        self.jtphi = jit(phi.apply)
      

        def _smart_init(x):
            module = xent_phi(archi, act)
            return module.smart_init(x)
        
        self.smart_init = hk.transform(_smart_init)

        @jit
        def batched_smart_init(params, batched_x):
            return vmap(self.smart_init.apply,
                        in_axes=(None, None, 0))(params, None, batched_x)
        
        self.batched_smart_init = batched_smart_init

        @jit
        def parallel_smart_init(params, par_x):
            return pmap(self.batched_smart_init)(params, par_x)

        self.parallel_smart_init = parallel_smart_init
  




    def init_params(self, in_size, in_chan):
        key = random.PRNGKey(self.seed)
        dum_x = jnp.ones((in_size, in_size, in_chan)) 
        dum_h = self.init_neurons(dum_x)
        dum_y = jnp.zeros(self.n_targets)

        params = self.phi.init(key, dum_x, dum_h, dum_y, 0.0) 
        if self.n_devices>1:
            params = tree_map(lambda x: jnp.array([x]*self.n_devices), params)
        return params




    @partial(jit, static_argnums=0)
    def init_neurons(self, x):
        size = x.shape[0] # x is HWC 
        h = []
    
        for i in range(self.archi['convlen']):
    
            l_name = 'xent_phi/~/conv_'+str(i+1) 
    
            c = self.archi[l_name]['channel']
            k = self.archi[l_name]['kernel']
            s = self.archi[l_name]['stride']
            p = self.archi[l_name]['pooling']
    
            if self.archi[l_name]['padding']=='SAME':
                pd = int((k-1)/2)
            elif self.archi[l_name]['padding']=='VALID':
                pd = 0
            size = int((size + 2*pd - k)/s +1)
            if p!='i':
                size = int( (size - 2)/2 + 1 )
            h.append(jnp.zeros((size, size, c)))
    
        for i in range(self.archi['fclen']-1):
            l_name = 'xent_phi/~/fc_'+str(i+1+self.archi['convlen'])
            s = self.archi[l_name]['fc']
            h.append(jnp.zeros(s))
    
        return tuple(h)

    @partial(jit, static_argnums=0)
    def batched_init_neurons(self, bat_x):
        out_ax = self.n_layers * (0,)
        btchd = vmap(self.init_neurons, in_axes=0, out_axes=out_ax)(bat_x)
        return btchd

    @partial(jit, static_argnums=0)
    def parallel_init_neurons(self, par_x):
        out_ax = self.n_layers * (0,)
        parl = pmap(self.batched_init_neurons, in_axes=0, out_axes=out_ax)(par_x)
        return parl









    @partial(jit, static_argnums=(0,4))
    def fwd(self, params, x, h, T, beta, y):
    
        holo = (x.dtype == jnp.complex64)

        for t in range(T):
            h, logits = grad(self.jtphi, has_aux=True, argnums=3,
                             holomorphic=holo)(params, None, x, h, y, beta)
            h = tree_map(self.act, h) 
        
        return h, logits

    @partial(jit, static_argnums=(0,4))
    def batched_fwd(self, *args):
        out_ax = self.n_layers * (0,)
        in_ax = (None, 0, out_ax, None, None, 0)
        batched = vmap(self.fwd, in_axes=in_ax, out_axes=(out_ax, 0))(*args)
        return batched

    @partial(jit, static_argnums=(0,4))
    def parallel_fwd(self, *args):
        out_ax = self.n_layers * (0,)
        in_ax = (0, 0, out_ax, None, None, 0)
        parl = pmap(self.batched_fwd, static_broadcasted_argnums=3,
                    in_axes=in_ax, out_axes=(out_ax, 0))(*args)
        return parl


    @partial(jit, static_argnums=(0,4))
    def fwd_for_fractal(self, params, x, h, T, beta, y):
    
        holo = (x.dtype == jnp.complex64)
        h0 = h

        for t in range(T):
            h, _ = grad(self.jtphi, argnums=3, has_aux=True,
                        holomorphic=holo)(params, None, x, h, y, beta)
            h = tree_map(self.act, h) 

        l2 = self.eval_conv(params, x, y, h, beta, True) 
        dist_h0 = L2(tree_map(lambda a, b: a-b, h0, h))
        l2 =  jnp.nan_to_num(l2, nan=1e4)
        dist_h0 =  jnp.nan_to_num(dist_h0, nan=1e4)

        return l2, dist_h0
         

    @partial(jit, static_argnums=(0,4))
    def batched_fwd_for_fractal(self, *args):
        in_ax = (None, None, None, None, 0, None)
        return vmap(self.fwd_for_fractal, in_axes=in_ax)(*args)






    @partial(jit, static_argnums=0)
    def _dEdw(self, params, x, h, beta, y):
        h = tree_map(lambda e: jnp.nan_to_num(e), h)
        holo = (x.dtype == jnp.complex64)
        dphidw, _ = grad(self.jtphi, has_aux=True, 
                         holomorphic=holo)(params, None, x, h, y, beta) 
        return div_param_dict(dphidw, -beta) # -beta because dEdw = -dphidw


    @partial(jit, static_argnums=0)
    def _batched_dEdw(self, *args):
        h_ax = self.n_layers*(0,)
        in_ax = (None, None, h_ax, 0, None)
        out_ax = dict_none_like(args[0])

        return vmap(lambda *args: psum(self._dEdw(*args), axis_name='j'),
                    in_axes=in_ax, out_axes=out_ax, axis_name='j')(*args)


    @partial(jit, static_argnums=(0,4))
    def fwd_for_grad(self, params, x, h, T, beta, y):
    
        holo = (x.dtype == jnp.complex64)

        for t in range(T):
            h, _ = grad(self.jtphi, argnums=3, has_aux=True,
                        holomorphic=holo)(params, None, x, h, y, beta)
            h = tree_map(self.act, h) 

        return h
         

    @partial(jit, static_argnums=(0,4))
    def batched_fwd_for_grad(self, *args):
        in_ax = (None, None, None, None, 0, None)
        return vmap(self.fwd_for_grad, in_axes=in_ax)(*args)





    @partial(jit, static_argnums=(0,4))
    def ep_grad1(self, params, x, h1, T2, beta, y):
        
        h2, _ = self.fwd(params, x, h1, T2, beta, y)
        dphi_1, _ = grad(self.jtphi, has_aux=True)(params, None, x, h1, y, 0.0)
        dphi_2, _ = grad(self.jtphi, has_aux=True)(params, None, x, h2, y, beta)
    
        grads = tree_map(lambda a, b: a-b, dphi_1, dphi_2)
        grads = div_param_dict(grads, beta)
    
        return grads




    @partial(jit, static_argnums=(0,4,5))
    def ep_gradN(self, params, x, h1, T, N, beta, y): 

        if N==1:
            grad = self.ep_grad1(params, x, h1, T, beta, y)
        else:
            if N%2==0:
                betas = jnp.array([beta, -beta])
                hs = self.batched_fwd_for_grad(params, x, h1, T, betas, y)
                grad = self._batched_dEdw(params, x, hs, betas, y) 
            else:
                h, _ = self.fwd(params, x, h1, T, beta, y)
                grad = self._dEdw(params, x, h, beta, y) 

            if N>2:
                cparams = to_complex_dict(params)
                h1 = to_complex(h1)
                x = x.astype(jnp.complex64)
   
                remain = (N+1)//2 - 1 
                first = 2*jnp.pi/N
                last = jnp.pi*( 1 - (2-(N%2))/N )
                angles = jnp.linspace(first, last, num=remain)
                betas = beta * jnp.exp(1j*angles)

                hs = self.batched_fwd_for_grad(cparams, x, h1, T, betas, y)

                l2s = self.batched_eval_conv_for_grad(cparams, x, y, hs, betas, True)
                safe = jnp.where(jnp.mean(jnp.log10(l2s))<=0, x=1.0, y=0.0)

                grad2 = self._batched_dEdw(cparams, x, hs, betas, y) 
                grad2 = to_2real_dict(grad2)
                grad2 = mul_param_dict(grad2, safe)
                grad = add_param_dict(grad, grad2)
                N = safe*N + (1-safe)*2
    
            grad = div_param_dict(grad, N)

        grad = tree_map(lambda e: jnp.nan_to_num(e), grad)

        return grad


    @partial(jit, static_argnums=(0,4,5))
    def batched_ep_gradN(self, *args):

        h_ax = self.n_layers*(0,)
        in_ax = (None, 0, h_ax, None, None, None, 0)
        out_ax = dict_none_like(args[0])

        return vmap(lambda *args: pmean(self.ep_gradN(*args), axis_name='i'),
                    in_axes=in_ax, out_axes=out_ax, axis_name='i')(*args)





    @partial(jit, static_argnums=(0,4,5))
    def slow_ep_gradN(self, params, x, h1, T, N, beta, y): 

        if N==1:
            grads = self.ep_grad1(params, x, h1, T, beta, y)
        else:
            grads = dict_zeros_like(params) 
            h, _ = self.fwd(params, x, h1, T, beta, y)
            inst_grads = self._dEdw(params, x, h, beta, y)
            grads = add_param_dict(grads, inst_grads)
    
            if N%2==0:
                h, _ = self.fwd(params, x, h1, T, -beta, y)
                inst_grads = self._dEdw(params, x, h, -beta, y)
                grads = add_param_dict(grads, inst_grads)
    
            if N>2:
                params = to_complex_dict(params)
                h1 = to_complex(h1)
                x = x.astype(jnp.complex64)
    
            for t in range(1, (N+1)//2): 
    
                cbeta = beta * jnp.exp((2 * 1j * jnp.pi * t)/N)
                h, _ = self.fwd(params, x, h1, T, cbeta, y)

                l2 = self.eval_conv(params, x, y, h, cbeta, True)
                safe = jnp.where(jnp.log10(l2)<=0, x=1.0, y=0.0)

                inst_grads = self._dEdw(params, x, h, cbeta, y)
                inst_grads = mul_param_dict(inst_grads, safe)
                inst_grads = to_2real_dict(inst_grads)
                grads = add_param_dict(grads, inst_grads)
     
            grads = div_param_dict(grads, N)

        grads = tree_map(lambda e: jnp.nan_to_num(e), grads)

        return grads

    @partial(jit, static_argnums=(0,4,5))
    def batched_slow_ep_gradN(self, *args):

        h_ax = self.n_layers*(0,)
        in_ax = (None, 0, h_ax, None, None, None, 0)
        out_ax = dict_none_like(args[0])

        return vmap(lambda *args: pmean(self.slow_ep_gradN(*args), axis_name='i'),
                    in_axes=in_ax, out_axes=out_ax, axis_name='i')(*args)





    def batched_loss(self, params, x, h, T, y):
        h1, logits = self.batched_fwd(params, x, h, T, 0.0, y)
        loss = -y*log_softmax(logits, axis=1)
        loss = jnp.sum(loss, axis=1)
        return jnp.mean(loss, axis=0), (logits, h1)






    @partial(jit, static_argnums=0)
    def sgd(self, params, grads, lrs, wds, mmt):
        
        mu = 0.9
        new_params = to_mutable_dict(params)

        for l in new_params:
            for p in new_params[l]:

                mmt[l][p] = mu * mmt[l][p] + grads[l][p]
                new_params[l][p] = \
                    (1-lrs[l]*wds[l]) * new_params[l][p] - lrs[l]*mmt[l][p]
        
        new_params = to_immutable_dict(new_params)  
    
        return new_params, mmt


    def drop_notconv(self, params, x, h, y, beta):
        holo = (x.dtype == jnp.complex64)
        L2s = self.batched_eval_conv(params, x, y, h, beta, holo)
        h = tree_map(lambda a: nanify(a, L2s), h)
        x = nanify(x, L2s)
        y = nanify(y, L2s)
        return x, h, y




    def batched_train_step(self, params, batch, Ts, beta, lrs, wds, mmt, algo):
        
        x, y = batch
        labels = y 
        y = one_hot(y, self.n_targets)
        T1, T2, N = Ts

        h = self.batched_init_neurons(x)

        if algo=='EP':
            h1, logits = self.batched_fwd(params, x, h, T1, 0.0, y)
            x, h11, yy = self.drop_notconv(params, x, h1, y, 0.0)
            grads = self.batched_ep_gradN(params, x, h11, T2, N, beta, yy)
        elif algo=='BPTT':        
            grads, aux = grad(self.batched_loss, 
                              has_aux=True)(params, x, h, T1, y)
            grads = tree_map(lambda e: jnp.nan_to_num(e), grads)
            logits, h1 = aux
   
 
        loss = -y*log_softmax(logits, axis=1)
        loss = jnp.sum(loss)
        pred = jnp.argmax(logits, axis=1)
        corr = jnp.sum(pred == labels)
        drop = jnp.isnan(yy).sum()/self.n_targets

        if self.n_devices>1:
            grads = pmean(grads, axis_name='d')
            corr = psum(corr, axis_name='d')
            loss = psum(loss, axis_name='d')
            drop = psum(drop, axis_name='d')

        params, mmt = self.sgd(params, grads, lrs, wds, mmt)
    
        return params, mmt, h1, corr, loss, drop


    def par_train_step(self, *args):
        in_ax = (0, (0, 0), (None,None,None), None, None, None, 0, None)
        return pmap(self.batched_train_step, axis_name='d', in_axes=in_ax, 
                    static_broadcasted_argnums=(2,7))(*args) 


    @partial(jit, static_argnums=(0,3,8))
    def train_step(self, *args):
        if self.n_devices>1:
            pbatch = split_batch(args[1])
            args = args[0], pbatch, *args[2:]
            return self.par_train_step(*args)
        else:
            return self.batched_train_step(*args)







    def eval_conv(self, params, x, y, h, beta, holo):
        h1, _ = grad(self.jtphi, argnums=3, has_aux=True,
                     holomorphic=holo)(params, None, x, h, y, beta)
        h1 = tree_map(self.act, h1) 
        h2, _ = grad(self.jtphi, argnums=3, has_aux=True,
                         holomorphic=holo)(params, None, x, h1, y, beta)
        h2 = tree_map(self.act, h2) 
        delta = tree_map(lambda a, b: a-b, h2, h)
        return L2(delta, cplx=holo)

    def batched_eval_conv(self, *args):
        h_ax = self.n_layers * (0,)
        in_ax = (None, 0, 0, h_ax, None, None)
        batched = vmap(self.eval_conv, in_axes=in_ax)(*args)
        return batched

    def batched_eval_conv_for_grad(self, *args):
        h_ax = self.n_layers * (0,)
        in_ax = (None, None, None, h_ax, 0, None)
        batched = vmap(self.eval_conv, in_axes=in_ax)(*args)
        return batched

    def parallel_eval_conv(self, *args):
        h_ax = self.n_layers * (0,)
        in_ax = (0, 0, 0, h_ax, None, None)
        parl = pmap(self.batched_eval_conv, in_axes=in_ax,
                    static_broadcasted_argnums=5)(*args)
        return parl

    @partial(jit, static_argnums=(0,6))
    def evaluate_convergence(self, *args):
        if args[5]:
            cparam = to_complex_dict(args[0])
            cx = args[1].astype(jnp.complex64)
            args = cparam, cx, *args[2:]
        if self.n_devices>1:
            px, py = split(args[1]), split(args[2])
            py = par_one_hot(py, self.n_targets)
            args = args[0], px, py, *args[3:]
            return self.parallel_eval_conv(*args)
        else:
            y = one_hot(args[2], self.n_targets)
            args = args[0], args[1], y, *args[3:]
            return self.batched_eval_conv(*args)








    @partial(jit, static_argnums=(0,4))
    def evaluate_batch(self, params, x, y, T):

        if self.n_devices == 1:
            size = x.shape[0]
            label = y
            y = one_hot(y, self.n_targets)

            h = self.batched_init_neurons(x)
            _, logits = self.batched_fwd(params, x, h, T, 0.0, y)

            loss = -y*log_softmax(logits, axis=1)
            loss = jnp.sum(loss)

            pred = jnp.argmax(logits, axis=1)
            correct = jnp.sum(jnp.equal(pred, label))

            pred5 = top_k(logits, 5)[1] 
            corr5 = jnp.sum(jnp.equal(pred5,
                            label.reshape(-1,1)).any(axis=1))
        else:
            size = x.shape[0]
            label = y
            y = one_hot(y, self.n_targets)

            x, y, label = split(x), split(y), split(label)
    
            h = self.parallel_init_neurons(x)
            _, logits = self.parallel_fwd(params, x, h, T, 0.0, y)

            loss = -y*log_softmax(logits, axis=2)
            loss = jnp.sum(loss)

            pred = jnp.argmax(logits, axis=2)
            correct = jnp.sum(jnp.equal(pred,label))

            pred5 = top_k(logits, 5)[1] 
            corr5 = jnp.sum(jnp.equal(pred5,
                         label.reshape(self.n_devices, -1,1)).any(axis=2))

        return loss, correct, corr5, size


    def evaluate(self, params, loader, T):
    
        tot_correct = 0
        tot_corr5 = 0
        sum_loss = 0
        tot_size = 0

        for x, y in loader:
            loss, correct, corr5, size = self.evaluate_batch(params, x, y, T)
            tot_correct += correct.item()
            tot_corr5 += corr5.item()
            sum_loss += loss.item()
            tot_size += size.item()
        
        acc = tot_correct/tot_size
        acc5 = tot_corr5/tot_size
        mean_loss = sum_loss/tot_size

        return acc, acc5, tot_correct, tot_corr5, tot_size, mean_loss


    def print_model(self, in_size, in_chan, path):
        dum_x = jnp.ones((in_size, in_size, in_chan)) 
        dum_h = self.init_neurons(dum_x)

        printLog('Input   :\t', dum_x.shape, path_to_result=path)
        for i, dh in enumerate(dum_h):
            printLog('Layer '+str(i+1)+' :\t', dh.shape, path_to_result=path)
        printLog('', path_to_result=path)
        







