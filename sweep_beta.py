import jax.numpy as jnp
import haiku as hk
import json
import pickle
import sys

from jax import grad, random

from utils.data import tfds_mnist_mlp, load_model, get_tfds_loaders
from utils.data import printLog, save_data, make_result_dir, one_hot
from utils.data import load_checkpoint
from utils.functions import get_activation
from utils.functions import to_complex, to_complex_dict
from utils.functions import init_cos_dist, update_cos_dist


# the first arg is the path to the json containing the hyperparameters
if len(sys.argv)>=2:
    with open(sys.argv[1]) as f:
        args = json.load(f)

path = make_result_dir(args, sys.argv[0]) if args['save'] else None


seed = args['seed']
key = random.PRNGKey(seed)
task = args['task']
bs = args['batch_size']
archi = hk.data_structures.to_immutable_dict(args['archi'])
beta = args['beta']
T1, T2 = args['T1'], args['T2']
Ns = args['N']
activation = get_activation(args)


if args['model_type']=='xp_mlp':

    from models import mlp
    
    n_targets = args['n_targets']

    noise = args['noise']
    model = mlp.xp_mlp(archi, activation, n_targets, seed, noise=noise)
    
    in_size = args['in_size']
    params = model.init_params(in_size)
    
    if args.get('load_model') is not None:
        name = args['load_model']
        params = load_model(params, name, path)
    
    if task=='mnist':    
        train_loader, _, _ = tfds_mnist_mlp(bs, seed)
        x, y = next(iter(train_loader))
    
    elif task=='toy':
        key = random.PRNGKey(seed)
        x = random.normal(key, (bs, in_size))
        x = x/x.max()
        key, subkey = random.split(key)
        y = random.randint(subkey, (bs,), 0, n_targets)

elif args['model_type']=='xent_p_cnn':

    from models import cnn
    
    n_devices = 1 

    train_loader, test_loader, n_targets, in_size, in_chan = \
        get_tfds_loaders(task, bs, seed)

    x, y = next(iter(train_loader))

    model = cnn.xent_p_cnn(archi, activation, n_targets, seed, n_devices)

    model.print_model(in_size, in_chan, path)
    params = model.init_params(in_size, in_chan)

    # if a model name has been specified in the command line
    if args.get('load_model') is not None:
        name = args['load_model']
        params = load_model(params, name, path)
        params, _, _, _ = load_checkpoint(params, name, path, n_devices)
   

y = one_hot(y, n_targets)
h = model.batched_init_neurons(x)  




if T1>300: # truncated bptt to avoid OOM
    h_int, _ = model.batched_fwd(params, x, h, T1-T2, 0.0, y)
    true_grads, aux = \
        grad(model.batched_loss, has_aux=True)(params, x, h_int, T2, y)
else:
    true_grads, aux = \
        grad(model.batched_loss, has_aux=True)(params, x, h, T1, y)
h0 = aux[1]





betas = jnp.linspace(0.01, beta, num=20)
# betas = [0.01, 0.1, 1.0]
# betas = 10**jnp.linspace(-4, 2, num=50)

EPs = {n: init_cos_dist(params) for n in Ns}


h1, _ = model.batched_fwd(params, x, h, T1, 0.0, y)

printLog(model.batched_eval_conv(params, x, y, h1, 0.0, False))




if args['model_type']=='xp_mlp':
    for bt in betas:
        for n in Ns:
            EP_est, key = \
                model.batched_ep_gradN(params, x, h1, T2, n, bt, y, key)
            EPs[n] = update_cos_dist(EPs[n], true_grads, EP_est)
else:
    for bt in betas:
        for n in Ns:
            EP_est = model.batched_ep_gradN(params, x, h1, T2, n, bt, y)
            EPs[n] = update_cos_dist(EPs[n], true_grads, EP_est)



data = {'x': betas, 'path': path, 'cos': {}, 'dist': {}}

for l in list(params.keys()) + ['total']:

    name = l.split('/')[-1]

    data_cos = [EPs[n][l]['cos'] for n in Ns]
    data_dist = [EPs[n][l]['dist'] for n in Ns]

    data['cos'][name] = {n: v for n, v in zip(Ns, data_cos)}
    data['dist'][name] = {n: v for n, v in zip(Ns, data_dist)}
    
    
save_data(data, path, 'data')


