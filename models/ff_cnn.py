import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, lax, pmap
from jax.nn import log_softmax
from jax.lax import pmean, psum, top_k
from jax import random
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
import matplotlib.pyplot as plt
import haiku as hk

from functools import partial
from utils.data import split, one_hot
from utils.custom import SfmPool



class Identity(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        return x



class cnn_forward_pass(hk.Module):
    def __init__(self, archi, activation, name=None):
        super().__init__(name=name)
  
        self.act = activation
        self.convlen = archi['convlen']        
        self.fclen = archi['fclen']

        layers = []
        pools = []
        for i in range(self.convlen):

            l_name = 'cnn_forward_pass/~/conv_'+str(i+1)

            c = archi[l_name]['channel']
            k = archi[l_name]['kernel']
            s = archi[l_name]['stride']
            pd = archi[l_name]['padding']

            if archi[l_name].get('sc') is not None:
                sc = archi[l_name]['sc']
                ini = hk.initializers.VarianceScaling(scale=sc, mode='fan_avg')
                layers.append(hk.Conv2D(c, k, stride=s, padding=pd,
                                        w_init=ini, name='conv_'+str(i+1)))
            else:
                layers.append(hk.Conv2D(c, k, stride=s, padding=pd,
                                        name='conv_'+str(i+1)))

            p = archi[l_name]['pooling']
            if p=='m':
                pools.append(hk.MaxPool((2,2,1), (2,2,1), padding='VALID'))
            elif p=='i':
                pools.append(Identity())
            elif p=='a':
                pools.append(hk.AvgPool((2,2,1), (2,2,1), padding='VALID'))
            elif p=='s':
                pools.append(SfmPool((2,2,1), (2,2,1), padding='VALID', T=1.0))
                    
        for i in range(self.fclen): 

            l_name = 'cnn_forward_pass/~/fc_'+str(i+1+self.convlen)
            size = archi[l_name]['fc']

            if archi[l_name].get('sc') is not None:
                sc = archi[l_name]['sc']
                ini = hk.initializers.VarianceScaling(scale=sc, mode='fan_avg')
                layers.append(hk.Linear(size, w_init=ini,
                                        name='fc_'+str(i+1+self.convlen)))
            else:
                layers.append(hk.Linear(size, name='fc_'+str(i+1+self.convlen)))

        self.layers = tuple(layers)
        self.pools = tuple(pools)

    def __call__(self, x):
        
        for i in range(self.convlen):
            m = self.layers[i]
            p = self.pools[i]
            x = self.act( p( m(x) ) )

        x = x.flatten()

        for i in range(self.convlen, len(self.layers)-1): # remove -1 for mse
            m = self.layers[i]
            x = self.act( m(x) )

        # comment out if not xent
        m = self.layers[-1]
        x = m(x)

        return x - logsumexp(x) # just x if not xent


class ff_cnn:

    def __init__(self, archi, act, n_targets, seed, n_devices):

        self.seed = seed
        # self.loss = loss
        self.n_targets = n_targets
        self.n_devices = n_devices

        def _fwd(x):
            module = cnn_forward_pass(archi, act)
            return module(x)
        
        self.fwd = hk.transform(_fwd)

        @jit
        def batched_fwd(params, batched_x):
            return vmap(self.fwd.apply,
                        in_axes=(None, None, 0))(params, None, batched_x)
        
        self.jtfwd = jit(self.fwd.apply)

        self.batched_fwd = batched_fwd

        @jit
        def parallel_fwd(params, par_x):
            return pmap(self.batched_fwd)(params, par_x)

        self.parallel_fwd = parallel_fwd

    def init_params(self, in_size, in_chan):

        dum_x = jnp.ones((in_size, in_size, in_chan))
        params = self.fwd.init(random.PRNGKey(self.seed), dum_x)
        if self.n_devices>1:
            params = tree_map(lambda x: jnp.array([x]*self.n_devices), params)

        return params

    @partial(jit, static_argnums=0)
    def batched_loss(self, params, x, y):
        x = self.batched_fwd(params, x)
        labels = y
        y = one_hot(y, self.n_targets)
        # if self.loss == 'xent':
        tot_loss = - jnp.sum(x * y, axis=1)
        loss = jnp.mean(tot_loss)
        tot_loss = jnp.sum(tot_loss)
        # elif self.loss == 'mse':
            # loss = 0.5 * jnp.mean(jnp.sum(jnp.power(x - y, 2), axis=1))

        pred = jnp.argmax(x, axis=1)
        corr = jnp.sum(jnp.equal(pred, labels))

        pred5 = top_k(y, 5)[1] 
        corr5 = jnp.sum(jnp.equal(pred5, labels.reshape(-1,1)).any(axis=1))

        return loss, (corr, corr5, tot_loss)


    @partial(jit, static_argnums=0)
    def sgd(self, params, grads, lrs, wds, mmt):
        
        mu = 0.9
        new_params = hk.data_structures.to_mutable_dict(params)

        for l in new_params:
            for p in new_params[l]:

                mmt[l][p] = mu * mmt[l][p] + grads[l][p]
                new_params[l][p] = \
                    (1-lrs[l]*wds[l]) * new_params[l][p] - lrs[l]*mmt[l][p]
        
        new_params = hk.data_structures.to_immutable_dict(new_params)  
    
        return new_params, mmt


    def batched_train_step(self, params, x, y, lrs, wds, mmt):
    
        grads, aux = grad(self.batched_loss, has_aux=True)(params, x, y)
        corr, corr5, tot_loss = aux

        if self.n_devices>1:
            grads = pmean(grads, axis_name='d')
            corr = psum(corr, axis_name='d')
            corr5 = psum(corr5, axis_name='d')
            tot_loss = psum(tot_loss, axis_name='d')
        
        params, mmt = self.sgd(params, grads, lrs, wds, mmt)
 
        return params, mmt, corr, corr5, tot_loss


    def par_train_step(self, *args):
        in_ax = (0, 0, 0, None, None, 0)
        return pmap(self.batched_train_step, 
                    axis_name='d', in_axes=in_ax)(*args) 


    @partial(jit, static_argnums=0)
    def train_step(self, *args):
        if self.n_devices>1:
            px, py = split(args[1]), split(args[2])
            args = args[0], px, py, *args[3:]
            return self.par_train_step(*args)
        else:
            return self.batched_train_step(*args)



    def evaluate(self, params, loader):
        
        correct = 0
        corr5 = 0
        size = 0
        loss = 0
        if self.n_devices>1:
            for x, y in loader:
                size += x.shape[0]
                label = y
                y = one_hot(y, self.n_targets)
                x, y, label = split(x), split(y), split(label)

                x = self.parallel_fwd(params, x)

                loss += - jnp.sum(x * y).item()

                pred = jnp.argmax(x, axis=2)
                correct += jnp.sum(jnp.equal(pred, label)).item()

                pred5 = top_k(x, 5)[1] 
                corr5 += jnp.sum(jnp.equal(pred5,
                         label.reshape(self.n_devices, -1,1)).any(axis=2)).item()
        else:
            for x, y in loader:
                size += x.shape[0]
                label = y
                y = one_hot(y, self.n_targets)

                x = self.batched_fwd(params, x)

                loss += - jnp.sum(x * y).item()

                pred = jnp.argmax(x, axis=1)
                correct += jnp.sum(jnp.equal(pred, label)).item()

                pred5 = top_k(x, 5)[1] 
                corr5 += jnp.sum(jnp.equal(pred5,
                                 label.reshape(-1,1)).any(axis=1)).item()

        
        acc = correct/size
        acc5 = corr5/size
        loss = loss/size

        return acc, acc5, correct, corr5, size, loss




