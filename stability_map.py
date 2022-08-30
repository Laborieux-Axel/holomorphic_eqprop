import jax.numpy as jnp
import numpy as np
import haiku as hk
import json
import pickle
import sys

from jax import random, local_device_count
from jax.tree_util import tree_map

from utils.data import tfds_mnist_mlp, load_model, get_tfds_loaders
from utils.data import printLog, save_data, make_result_dir, one_hot
from utils.data import load_checkpoint
from utils.functions import get_activation, to_complex, L2, to_complex_dict

# the first arg is the path to the json containing the hyperparameters
if len(sys.argv)>=2:
    with open(sys.argv[1]) as f:
        args = json.load(f)

path = make_result_dir(args, sys.argv[0]) if args['save'] else None


if args['model_type']=='xp_mlp':
    
    from models import mlp
    
    # Unpacking argument values    
    seed = args['seed']
    batch_size = args['batch_size']
    in_size = args['in_size']
    n_targets = args['n_targets']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    T1, T2 = args['T1'], args['T2']
    Ts = T1, T2
    activation = get_activation(args)
    
    
    if args['task']=='mnist':    
        train_loader, _, _ = tfds_mnist_mlp(batch_size, seed)
        batch = next(iter(train_loader))
        x, y = batch

    elif args['task']=='toy':
        key = random.PRNGKey(seed)
        x = random.normal(key, (batch_size, in_size))
        x = x/x.max()
        key, subkey = random.split(key)
        y = random.randint(subkey, (batch_size,), 0, n_targets)
        print(x)
        print(y)
    
    model = mlp.xp_mlp(archi, activation, n_targets, seed)

    params = model.init_params(in_size)
    
    # if a model name has been specified in the command line
    if args.get('load_model') is not None:
        name = args['load_model']
        params = load_model(params, name, path)
     

    h = model.batched_init_neurons(x)  
    y = one_hot(y, n_targets)

    
    h1, _ = model.batched_fwd(params, x, h, T1, 0.0, y)

    L2s = model.batched_eval_conv(params, x, y, h1, 0.0, False)
    printLog(L2s, path_to_result=path)
    
    neural_data = {'free_eq': h1, 'path': path}
    save_data(neural_data, path, 'neural_data')
    
    # Complex phase
    cparams = to_complex_dict(params)
    h1 = to_complex(h1)
    x = x.astype(jnp.complex64)

    
    a0 = args['xcenter'] - 0.5*args['width']
    a1 = args['xcenter'] + 0.5*args['width']
    b0 = args['ycenter'] - 0.5*args['width']
    b1 = args['ycenter'] + 0.5*args['width']

    irx, iry = args['resol'], args['resol'] # resolution of output
    
    Re_betas = jnp.linspace(a0, a1, irx+1).reshape(1, -1)
    Im_betas = jnp.linspace(b0, b1, iry+1).reshape(-1, 1)
    betas = Re_betas + 1j*Im_betas
    betas = betas.flatten()
    z, _ = model.batched_fwd_for_fractal(params, x, h1, T2, betas, y)
    z = z.reshape(iry+1, irx+1)
    z = jnp.clip(z, a_max=1e4)

    X, Y = np.meshgrid(np.array(Re_betas), np.array(Im_betas))

    data = {'dyn_stab': (X, Y, z),
            'xrange': (a0, a1),
            'yrange': (b0, b1),
            'path': path}

    save_data(data, path, 'data')






elif args['model_type']=='xent_p_cnn':

    from models import cnn
    
    # Unpacking argument values    
    seed = args['seed']
    task = args['task']
    batch_size = args['batch_size']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    T1, T2 = args['T1'], args['T2']
    Ts = T1, T2
    activation = get_activation(args)

    
    n_devices = local_device_count() if args['parallel'] else 1
    print(f"Number of devices: {n_devices}")
    
    
    train_loader, test_loader, n_targets, in_size, in_chan = \
        get_tfds_loaders(task, batch_size, seed)

    model = cnn.xent_p_cnn(archi, activation, n_targets, seed, n_devices)

    model.print_model(in_size, in_chan, path)
    params = model.init_params(in_size, in_chan)

    # if a model name has been specified in the command line
    if args.get('load_model') is not None:
        name = args['load_model']
        params = load_model(params, name, path)
        # params, _, _, _ = load_checkpoint(params, name, path, n_devices)
     
    
    x, y = next(iter(train_loader))
    h = model.batched_init_neurons(x)  
    y = one_hot(y, n_targets)
    
    
    h1, _ = model.batched_fwd(params, x, h, T1, 0.0, y)

    L2s = model.batched_eval_conv(params, x, y, h1, 0.0, False)

    printLog(L2s, path_to_result=path)


    # Complex phase
    cparams = to_complex_dict(params)
    h1 = to_complex(h1)
    x = x.astype(jnp.complex64)

    a0 = args['xcenter'] - 0.5*args['width']
    a1 = args['xcenter'] + 0.5*args['width']
    b0 = args['ycenter'] - 0.5*args['width']
    b1 = args['ycenter'] + 0.5*args['width']

    irx, iry = args['resol'], args['resol'] # resolution of output
    
    Re_betas = jnp.linspace(a0, a1, irx+1).reshape(1, -1)
    Im_betas = jnp.linspace(b0, b1, iry+1).reshape(-1, 1)
    betas = Re_betas + 1j*Im_betas
    betas = betas.flatten()

    z, _ = model.batched_fwd_for_fractal(cparams, x, h1, T2, betas, y)
    z = z.reshape(iry+1, irx+1)
    z = jnp.clip(z, a_max=1e4)

    X, Y = np.meshgrid(np.array(Re_betas), np.array(Im_betas))

    data = {'dyn_stab': (X, Y, z),
            'xrange': (a0, a1),
            'yrange': (b0, b1),
            'path': path}

    save_data(data, path, 'data')
    
