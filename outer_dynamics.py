import jax.numpy as jnp
import haiku as hk
import json
import sys

# from torch import manual_seed
from jax import grad, local_device_count
from jax import random, jit
from jax.tree_util import tree_map

from utils.data import tfds_mnist_mlp, load_model,  get_tfds_loaders
from utils.data import printLog, save_data, one_hot, load_checkpoint
from utils.data import make_result_dir, split, plot_img, par_one_hot, unsplit
from utils.functions import to_complex, L2, to_complex_dict, get_activation
from utils.functions import div_param_dict, dict_zeros_like, cosine_dicts
from utils.functions import distance_dicts


# the first arg is the path to the json containing the hyperparameters
if len(sys.argv)>=2:
    with open(sys.argv[1]) as f:
        args = json.load(f)


path = make_result_dir(args, sys.argv[0]) if args['save'] else None


if args['model_type']=='xp_mlp':
    
    from models import mlp
    
    # Unpacking argument values    
    seed = args['seed']
    key = random.PRNGKey(seed)
    batch_size = args['batch_size']
    in_size = args['in_size']
    n_targets = args['n_targets']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    beta = args['beta']
    noise = args['noise']
    T1, T2, N = args['T1'], args['T2'], args['N']
    activation = get_activation(args)
    
    
    if args['task']=='mnist':    
        train_loader, _, _ = tfds_mnist_mlp(batch_size, seed)
        batch = next(iter(train_loader))
        x, y = batch     

    elif args['task']=='toy':
        x = random.normal(key, (batch_size, in_size))
        x = x/x.max()
        key, subkey = random.split(key)
        y = random.randint(subkey, (batch_size,), 0, n_targets)
        print(x)
        print(y)
 
    model = mlp.xp_mlp(archi, activation, n_targets, seed, noise=noise)

    params = model.init_params(in_size)
    
    if args.get('load_model') is not None:
        name = args['load_model']
        params = load_model(params, name, path)
    
    if args['plot']:
        mlp.plot_params(params, path) 

    h = model.batched_init_neurons(x)  
    y = one_hot(y, n_targets)


    bptt_grads, aux = grad(model.batched_loss,
                           has_aux=True)(params, x, h, T1, y)
    _, h0 = aux


    printLog('Convergence for bptt grads:', path_to_result=path)
    printLog(model.batched_eval_conv(params, x, y, h0, 0.0, False).mean(), path_to_result=path)


    EP1, key = model.batched_ep_gradN(params, x, h0, N, 1, beta, y, key)
    cos_ep1 = cosine_dicts(bptt_grads, EP1)

    h = to_complex(h) # start from scratch
    cparams = to_complex_dict(params)
    x = x.astype(jnp.complex64)

    cbeta = beta * jnp.exp(- (2j * jnp.pi)/N)

    grads = dict_zeros_like(params)

    continual_cosine = []
    continual_dist = []
    continual_cosine.append(0.0)
    continual_dist.append(distance_dicts(bptt_grads, grads))

    printLog('Residual convergence after each period:')

    N_periods = 40
    mean_log_L2s = []

    for t in range(N_periods):
        h, grads, key = \
            model.batched_outer_fwd(cparams, x, h, T2, N, beta, y, grads, key)

        L2s = model.batched_eval_conv(params, x, y, h, cbeta, True).mean()
        mean_log_L2s.append(jnp.log10(L2s))
        printLog(jnp.log10(L2s), path_to_result=path)

        norm_grads = div_param_dict(grads, N*(t+1))
        continual_cosine.append(cosine_dicts(bptt_grads, norm_grads))
        continual_dist.append(distance_dicts(bptt_grads, norm_grads))


    printLog('Final cosine with grad:\t', continual_cosine[-1], path_to_result=path)
    printLog('Final dist with grad:\t', continual_dist[-1], path_to_result=path)

    data = {'cont_cos': continual_cosine,
            'cont_dist': continual_dist,
            'cos_ep1': cos_ep1,
            'mean_log_L2s': mean_log_L2s,
            'path': path}
    
    save_data(data, path, 'data')



