import jax.numpy as jnp
import numpy as np
import haiku as hk
from haiku.data_structures import to_mutable_dict, to_immutable_dict
from jax import grad, jit, vmap, lax
from functools import partial
from jax import random
from jax.nn import log_softmax
from jax.lax import pmean, cond
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

import os

from utils.data import one_hot
from utils.functions import dict_zeros_like, add_param_dict
from utils.functions import dict_none_like
from utils.functions import to_real_dict, div_param_dict, to_complex
from utils.functions import to_complex_dict
from utils.functions import to_2real_dict, L2





class xphi(hk.Module):
    def __init__(self, archi, name=None):
        super().__init__(name=name)
        
        layers = []
        for i in range(len(archi)):
            l_name = 'xphi/~/fc_'+str(i+1)
            size = archi[l_name]['fc']
            
            if archi[l_name].get('sc') is not None:
                ini = hk.initializers.TruncatedNormal(stddev=archi[l_name]['sc'])
                layers.append(hk.Linear(size, name='fc_'+str(i+1), w_init=ini))
            else:
                layers.append(hk.Linear(size, name='fc_'+str(i+1)))
        
        self.layers = tuple(layers)

    def __call__(self, x, h, y, beta):
        
        phi = 0.0
        l = x, *h

        for i in range(len(self.layers)-1):
            m = self.layers[i]
            phi = phi + jnp.sum( m( l[i] ) * l[i+1] )

        logits = self.layers[-1](l[-1])
        xent = -y*log_softmax(logits)

        phi = phi - beta * jnp.sum(xent)

        return phi, logits



class xp_mlp:

    def __init__(self, archi, act, n_targets, seed, noise=0.0):
       
        self.archi = archi
        self.act = act 
        self.n_targets = n_targets
        self.seed = seed
        self.noise_level = noise

        def _phi(x, h, y, beta):
            module = xphi(archi)
            return module(x, h, y, beta)
        
        phi_func = hk.transform(_phi)
        self.phi = phi_func
        self.jtphi = jit(phi_func.apply)




    def init_params(self, in_size):
        dum_x = random.normal(random.PRNGKey(1), (in_size,))
        dum_h = self.init_neurons(dum_x)
        dum_y = jnp.zeros(self.n_targets)
        params = \
             self.phi.init(random.PRNGKey(self.seed), dum_x, dum_h, dum_y, 0.0)
        return params        

    def init_neurons(self, x):
        h = []
        for i in range(len(self.archi)-1):
            s = self.archi['xphi/~/fc_'+str(i+1)]['fc']
            h.append(jnp.zeros(s))
        return tuple(h)
    
    def batched_init_neurons(self, bat_x):
        out_ax = (len(self.archi)-1)*(0,)
        return vmap(self.init_neurons, in_axes=0, out_axes=out_ax)(bat_x)



    def init_noise(self, key):
        n = []
        for i in range(len(self.archi)-1):
            s = self.archi['xphi/~/fc_'+str(i+1)]['fc']
            key, subkey = random.split(key) 
            n.append(self.noise_level * random.normal(subkey, (s,)))
        return tuple(n), key








    @partial(jit, static_argnums=(0,4))
    def fwd(self, params, x, h, T, beta, y):
    
        holo = (x.dtype == jnp.complex64)

        for t in range(T):
            h, logits = grad(self.jtphi, argnums=3, has_aux=True,
                             holomorphic=holo)(params, None, x, h, y, beta)
            h = tree_map(self.act, h) 
        
        return h, logits

    @partial(jit, static_argnums=(0,4))
    def batched_fwd(self, *args):
        out_ax = (len(self.archi)-1)*(0,)
        in_ax = (None, 0, out_ax, None, None, 0)
        return vmap(self.fwd, in_axes=in_ax, out_axes=(out_ax, 0))(*args)





    @partial(jit, static_argnums=(0,4))
    def noisy_fwd(self, params, x, h, T, beta, y, key):
    
        holo = (x.dtype == jnp.complex64)

        for t in range(T):
            h, logits = grad(self.jtphi, argnums=3, has_aux=True,
                             holomorphic=holo)(params, None, x, h, y, beta)
            noise, key = self.init_noise(key)
            noisy_h = tree_map(lambda a, b: a+b, h, noise)
            h = tree_map(self.act, noisy_h) 
        
        return h, logits, key

    @partial(jit, static_argnums=(0,4))
    def batched_noisy_fwd(self, *args):
        out_ax = (len(self.archi)-1)*(0,)
        in_ax = (None, 0, out_ax, None, None, 0, None)
        return vmap(self.noisy_fwd, in_axes=in_ax,
                    out_axes=(out_ax, 0, None))(*args)







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
    



    ## COMPLEX METHODS ## 


    @partial(jit, static_argnums=(0,4))
    def ep_grad1(self, params, x, h1, T2, beta, y):
        
        h2, _ = self.fwd(params, x, h1, T2, beta, y)
        dphi_1, _ = grad(self.jtphi, has_aux=True)(params, None, x, h1, y, 0.0)
        dphi_2, _ = grad(self.jtphi, has_aux=True)(params, None, x, h2, y, beta)
        
        grads = tree_map(lambda a, b: a-b, dphi_1, dphi_2)
        grads = div_param_dict(grads, beta)
        
        return grads


    @partial(jit, static_argnums=(0,4))
    def noisy_ep_grad1(self, params, x, h1, T2, beta, y, key):
        
        # dphi_1, _ = grad(self.jtphi, has_aux=True)(params, None, x, h1, y, 0.0)
        # dphi_2, _ = grad(self.jtphi, has_aux=True)(params, None, x, h2, y, beta)

        h1, _, key = self.noisy_fwd(params, x, h1, 1, 0.0, y, key)
        dphi_1_av, _ = grad(self.jtphi, has_aux=True)(params, None, x, h1, y, 0.0)
        h2, _, key = self.noisy_fwd(params, x, h1, T2, beta, y, key)
        dphi_2_av, _ = grad(self.jtphi, has_aux=True)(params, None, x, h2, y, beta)

        N = 5
        for i in range(N-1):
            h1, _, key = self.noisy_fwd(params, x, h1, 1, 0.0, y, key)
            dphi_1, _ = grad(self.jtphi, has_aux=True)(params, None, x, h1, y, 0.0)
            dphi_1_av = tree_map(lambda a, b: a+b, dphi_1_av, dphi_1)

            h2, _, key = self.noisy_fwd(params, x, h2, 1, beta, y, key)
            dphi_2, _ = grad(self.jtphi, has_aux=True)(params, None, x, h2, y, beta)
            dphi_2_av = tree_map(lambda a, b: a+b, dphi_2_av, dphi_2)

        dphi_1_av = div_param_dict(dphi_1_av, N)
        dphi_2_av = div_param_dict(dphi_2_av, N)
        
        grads = tree_map(lambda a, b: a-b, dphi_1_av, dphi_2_av)
        grads = div_param_dict(grads, beta)
        
        return grads, key



    @partial(jit, static_argnums=0)
    def _dEdw(self, params, x, h, beta, y):
        h = tree_map(lambda e: jnp.nan_to_num(e), h)
        holo = (x.dtype == jnp.complex64)
        dphidw, _ = grad(self.jtphi, has_aux=True, 
                         holomorphic=holo)(params, None, x, h, y, beta) 
        return div_param_dict(dphidw, -beta) # -beta because dEdw = -dphidw



    @partial(jit, static_argnums=(0,4,5))
    def noisy_ep_gradN(self, params, x, h, T, N, beta, y, key): 

        if N==1:
            grads, key = self.noisy_ep_grad1(params, x, h, T, beta, y, key)
        else:
            grads = dict_zeros_like(params) 
            h = to_complex(h)
            params = to_complex_dict(params)
            x = x.astype(jnp.complex64)
    
            for t in range(N):
    
                cbeta = beta * jnp.exp((2 * 1j * jnp.pi * t)/N)
                h, _, key = self.noisy_fwd(params, x, h, T, cbeta, y, key)
                inst_grads = self._dEdw(params, x, h, cbeta, y)
    
                grads = add_param_dict(grads, inst_grads)
     
            grads = div_param_dict(grads, N)
            grads = to_real_dict(grads)

        return grads, key







    @partial(jit, static_argnums=(0,4,5))
    def fast_ep_gradN(self, params, x, h1, T, N, beta, y, key): 

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
                inst_grads = self._dEdw(params, x, h, cbeta, y)
                inst_grads = to_2real_dict(inst_grads)
                grads = add_param_dict(grads, inst_grads)
     
            grads = div_param_dict(grads, N)

        return grads, key



    @partial(jit, static_argnums=(0,4,5))
    def batched_ep_gradN(self, *args):

        h_ax = (len(self.archi)-1)*(0,)
        in_ax = (None, 0, h_ax, None, None, None, 0, None)
        out_ax = (dict_none_like(args[0]), None)

        if self.noise_level==0.0: # in this case save time by symmetry
    
            @partial(vmap, in_axes=in_ax, out_axes=out_ax, axis_name='i')
            def f(*args):
                grads, key = self.fast_ep_gradN(*args)
                grads = pmean(grads, axis_name='i')
                return grads, key

            # grads, key = vmap(lambda *args: pmean(self.fast_ep_gradN(*args), axis_name='i'),
                         # in_axes=in_ax, out_axes=out_ax, axis_name='i')(*args)
        else:
    
            @partial(vmap, in_axes=in_ax, out_axes=out_ax, axis_name='i')
            def f(*args):
                grads, key = self.noisy_ep_gradN(*args)
                grads = pmean(grads, axis_name='i')
                return grads, key

        return f(*args)





    # @partial(jit, static_argnums=(0,4,5))
    def outer_fwd(self, params, x, h, T2, N, beta, y, grads, key):
   
        for t in range(N): 
            h, _, key = self.noisy_fwd(params, x, h, T2, beta, y, key)

            inst_grads = self._dEdw(params, x, h, beta, y)
            g_update = lambda a, b: a + jnp.real(b) 
            grads = tree_map(g_update, grads, inst_grads)

            beta = beta * jnp.exp((2j * jnp.pi)/N)
    
        return h, grads, key

    def batched_outer_fwd(self, *args):
        h_ax = (len(self.archi)-1)*(0,)
        in_ax = (None, 0, h_ax, None, None, None, 0, None, None)
        out_ax = (0, None, None)

        @partial(vmap, in_axes=in_ax, out_axes=out_ax, axis_name='i')
        def f(*args):
            h, grads, key = self.outer_fwd(*args)
            grads = pmean(grads, axis_name='i')
            return h, grads, key
       
        return f(*args) 


    # @partial(jit, static_argnums=(0,5,6,7))
    def batched_plasticity(self, params, x, h, y, T_plas, T_teach, T_dyn, beta, key):

        h = to_complex(h)
        grads = dict_zeros_like(params)
        cparams = to_complex_dict(params)
        x = x.astype(jnp.complex64)
    
        for t in range(T_plas):
            h, grads, key = \
                self.batched_outer_fwd(cparams, x, h, T_dyn, T_teach,
                                       beta, y, grads, key)

        # grads = to_real_dict(grads)
        grads = div_param_dict(grads, T_teach*T_plas)
        return grads, h, key




    @partial(jit, static_argnums=(0,4))
    def batched_loss(self, params, x, h, T, y):
        h1, logits = self.batched_fwd(params, x, h, T, 0.0, y)
        loss = -y*log_softmax(logits, axis=1)
        loss = jnp.sum(loss, axis=1)
        return jnp.mean(loss, axis=0), (logits, h1)








    @partial(jit, static_argnums=0)
    def sgd(self, params, grads, lrs):
        
        new_params = to_mutable_dict(params)
        
        for l in new_params:
            for p in new_params[l]:
                new_params[l][p] = new_params[l][p] - lrs[l] * grads[l][p]

        new_params = to_immutable_dict(new_params)
        return new_params


    @partial(jit, static_argnums=(0,3,6))
    def train_step(self, params, batch, Ts, beta, lrs, algo, key):
        
        x, y = batch
        y = one_hot(y, self.n_targets)
        T1, T2, N = Ts
        h = self.batched_init_neurons(x)
        
        if algo=='EP':
            h1, _ = self.batched_fwd(params, x, h, T1, 0.0, y)
            grads, key = self.batched_ep_gradN(params, x, h1, T2, N, beta, y, key)
        elif algo=='BPTT':
            grads, aux = grad(self.batched_loss, 
                              has_aux=True)(params, x, h, T1, y)
            _, h1 = aux
        elif algo=='oscEP':
            grads, h1, key = \
                self.batched_plasticity(params, x, h, y, 3, N, T2, beta, key)
            
        new_params = self.sgd(params, grads, lrs)
    
        return new_params, h1, key




    def eval_conv(self, params, x, y, h, beta, holo):
        h_next, _ = grad(self.jtphi, argnums=3, has_aux=True,
                         holomorphic=holo)(params, None, x, h, y, beta)
        h_next = tree_map(self.act, h_next) 
        delta = tree_map(lambda a, b: a-b, h_next, h)
        return L2(delta, cplx=holo)

    def batched_eval_conv(self, *args):
        h_ax = (len(self.archi)-1)*(0,)
        in_ax = (None, 0, 0, h_ax, None, None)
        batched = vmap(self.eval_conv, in_axes=in_ax)(*args)
        return batched



    def evaluate(self, params, loader, T):
        
        size = 0
        correct = 0
        loss = 0
        for x, y in loader:
            size += x.shape[0]
            label = y
            y = one_hot(y, self.n_targets)

            h = self.batched_init_neurons(x)
            _, logits = self.batched_fwd(params, x, h, T, 0.0, y)

            loss += jnp.sum(-y*log_softmax(logits, axis=1)).item()
            pred = jnp.argmax(logits, axis=1)
            correct += jnp.sum(jnp.equal(pred, label)).item()
        
        acc = correct/size
        loss = loss/size
        return acc, correct, size, loss


