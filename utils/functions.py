import jax.numpy as jnp
import haiku as hk
from jax import jit





def identity(x):
    return x

def silu(z):
    return z/(1 + jnp.exp(-z))

def my_sigmoid(a, b):
    def f(x):
        return 1/(1 + jnp.exp(a * x + b))
    return f

def my_hardtanh(x):
    return jnp.minimum(jnp.maximum(0, 0.5*x), 1)

def hardtanh(x):
    return jnp.minimum(jnp.maximum(0, x), 1)

def relu(x):
    return jnp.maximum(0, x)

def holotanh(a):
    def f(z):
        return z/(1 + jnp.exp(-a*z)) + (1-z)/(1 + jnp.exp(-a*(z-1)) )
    return f

def my_holotanh(a):
    def f(z):
        return 0.5*z/(1 + jnp.exp(-a*z)) + (1-0.5*z)/(1 + jnp.exp(-a*(z-2)) )
    return f

def my_cos(a):
    def f(x):
        return 0.5 * (1 - jnp.cos(a*jnp.pi*x))
    return f

def elu(a):
    def f(z):
	    return jnp.where(z>=0.0, x=z, y=a*(jnp.exp(z)-1))
    return f




def get_activation(args):
    if args['activation']=='my_sigmoid':
        a, b = args['a'], args['b']
        activation = my_sigmoid(a, b)
    elif args['activation']=='hardtanh':
        activation = hardtanh
    elif args['activation']=='my_hardtanh':
        activation = my_hardtanh
    elif args['activation']=='relu':
        activation = relu
    elif args['activation']=='holotanh':
        activation = holotanh(args['a'])
    elif args['activation']=='my_holotanh':
        activation = my_holotanh(args['a'])
    elif args['activation']=='silu':
        activation = silu
    elif args['activation']=='identity':
        activation = identity
    elif args['activation']=='my_cos':
        activation = my_cos(args['a'])
    elif args['activation']=='elu':
        activation = elu(args['a'])
    return activation




def init_lrs_wds_mmt(archi, params):
    lrs, init_lrs, final_lrs, wds, mmt = {}, {}, {}, {}, {}
    for l in params:
        init_lrs[l] = archi[l]['lr']
        lrs[l] = archi[l]['lr']
        final_lrs[l] = archi[l]['lr'] * 1e-7
        wds[l] = archi[l]['wd']
        mmt[l] = {}
        for p in params[l]:
            mmt[l][p] = jnp.zeros_like(params[l][p], dtype=jnp.float32)
    return lrs, init_lrs, final_lrs, wds, mmt


@jit
def cos_ann(lrs, init_lrs, final_lrs, t, t_max):
    for l in lrs:
        lrs[l] = ( final_lrs[l]
                   + 0.5 
                   * (init_lrs[l] - final_lrs[l])
                   * (1 + jnp.cos((t*jnp.pi)/t_max)) )
    return lrs


def init_cos_dist(params):
    out = {}
    for l in params:
        out[l] = {}
        out[l]['cos'] = []
        out[l]['dist'] = []
    out['total'] = {}
    out['total']['cos'] = []
    out['total']['dist'] = []
    return out

def update_cos_dist(track, grad, est):
    track['total']['cos'].append(cosine_dicts(grad, est))
    track['total']['dist'].append(distance_dicts(grad, est))
    for l in grad:
        grad_l, est_l = flatten_layer(grad[l]), flatten_layer(est[l])
        cos_l = jnp.inner(grad_l, est_l)/(jnp.linalg.norm(grad_l) 
                                          * jnp.linalg.norm(est_l))
        track[l]['cos'].append(cos_l.item())
        dist_l = jnp.linalg.norm(grad_l - est_l)
        track[l]['dist'].append(dist_l.item())
    return track


def check_nan_dict(d):
    s = 0.0
    for l in d:
        for p in d[l]:
            s += jnp.sum(d[l][p])
    return jnp.isnan(s).item()


def distance_dicts(grad, estimate):
    flat_grad, flat_est = flatten_dict(grad), flatten_dict(estimate)
    return jnp.linalg.norm(flat_grad - flat_est).item()

def cosine_dicts(a, b):
    a, b = flatten_dict(a), flatten_dict(b)
    cos = jnp.inner(a, b)/(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return cos.item()


def flatten_dict(a):
    concat_a = None
    for l in a:
        for p in a[l]:
            if concat_a is not None:
                concat_a = jnp.concatenate((concat_a, a[l][p].flatten()), axis=0)
            else:
                concat_a = a[l][p].flatten()
    return concat_a


def flatten_layer(l):
    concat_p = None
    for p in l:
        if concat_p is not None:
            concat_p = jnp.concatenate((concat_p, l[p].flatten()), axis=0)
        else:
            concat_p = l[p].flatten()
    return concat_p


def max_diff(a, b):
    curr_max = 0.0
    for l in a:
        for p in a[l]:
            new_max = jnp.amax(jnp.absolute(a[l][p] - b[l][p]))
            curr_max = new_max if new_max>curr_max else curr_max
    return curr_max


def flatten_list(a):
    out = None
    for aa in a:
        if out is not None:
            out = jnp.concatenate((out, aa.flatten()), axis=0)
        else:
            out = aa.flatten()
    return out


def cosine_lists(a, b):
    a, b = flatten_list(a), flatten_list(b)
    cos = jnp.inner(a, b)/(jnp.linalg.norm(a) * jnp.linalg.norm(b))
    return cos.item()


def L2(u, cplx=False):
    out = 0.0
    for uu in u:
        if cplx:
            out += jnp.real(jnp.sum(jnp.conjugate(uu)*uu))
        else:
            out += jnp.power(jnp.linalg.norm(uu), 2)
    return jnp.power(out + 1e-12, 0.5)



def L2_dict(p):
    return jnp.linalg.norm(flatten_dict(p))



@jit
def to_complex(h):
    new_h = []
    for e in h:
        new_h.append(e.astype(jnp.complex64))
    return tuple(new_h)

@jit
def to_conjugate(h):
    new_h = []
    for e in h:
        new_h.append(jnp.conjugate(e))
    return tuple(new_h)








@jit
def dict_zeros_like(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = jnp.zeros_like(d[l][p])
    return out


def dict_none_like(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = None
    return out

def dict_zero_like(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = int(0)
    return out

@jit
def add_param_dict(a, b):
    out = {}
    for l in a:
        out[l] = {}
        for p in a[l]:
            out[l][p] = a[l][p] + b[l][p]
    return out



@jit
def param_norm(a, cst):
    norm = L2_dict(a)
    out = {}
    for l in a:
        out[l] = {}
        for p in a[l]:
            out[l][p] = (a[l][p]*cst)/norm
    return out


@jit
def div_param_dict(a, cst):
    out = {}
    for l in a:
        out[l] = {}
        for p in a[l]:
            out[l][p] = a[l][p]/cst
    return out



@jit
def mul_param_dict(a, cst):
    out = {}
    for l in a:
        out[l] = {}
        for p in a[l]:
            out[l][p] = a[l][p] * cst
    return out


@jit
def to_real_dict(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = jnp.real(d[l][p])
    return out



@jit
def to_2real_dict(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = 2 * jnp.real(d[l][p])
    return out


@jit
def to_complex_dict(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = d[l][p].astype(jnp.complex64)
    return out

@jit
def to_module(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = jnp.absolute(d[l][p])
    return out

@jit
def to_conjugate_dict(d):
    out = {}
    for l in d:
        out[l] = {}
        for p in d[l]:
            out[l][p] = jnp.conjugate(d[l][p])
    return out



def nanify(a, l2):
    trailing_shape = (1,)*(len(a.shape)-1)
    l2 = l2.reshape((-1, *trailing_shape))
    out = jnp.where(jnp.log10(l2)<=2., x=a, y=jnp.nan)
    return out




