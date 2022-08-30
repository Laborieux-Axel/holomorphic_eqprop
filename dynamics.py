import jax.numpy as jnp
import haiku as hk
import json
import sys

from jax import grad, local_device_count, random
from jax.tree_util import tree_map

from utils.data import tfds_mnist_mlp, load_model, get_tfds_loaders
from utils.data import printLog, save_data, one_hot, load_checkpoint
from utils.data import make_result_dir, split, par_one_hot
from utils.functions import to_complex, L2, to_complex_dict, get_activation
from utils.functions import div_param_dict, dict_zeros_like, cosine_dicts


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
        key = random.PRNGKey(seed)
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
    

    h = model.batched_init_neurons(x)  
    y = one_hot(y, n_targets)


    h = model.batched_init_neurons(x)  
    true_grads, _ = grad(model.batched_loss, has_aux=True)(params, x, h, 30, y)


    dEds = jnp.zeros((batch_size, T1+1))
    h = model.batched_init_neurons(x)  
    L2s = model.batched_eval_conv(params, x, y, h, 0.0, False)
    dEds = dEds.at[:, 0].set(L2s)
    t, crit = 0, jnp.mean(jnp.log10(L2s)).item()
    
    while (t<T1) and (crit>=-5.8):
        if noise==0.0:
            h, _ = model.batched_fwd(params, x, h, 1, 0.0, y)
        else:
            h, _, key = model.batched_noisy_fwd(params, x, h, 1, 0.0, y, key)
        L2s = model.batched_eval_conv(params, x, y, h, 0.0, False)
        dEds = dEds.at[:, t+1].set(L2s)
        crit = jnp.mean(jnp.log10(L2s)).item()
        t+=1
    
    printLog(jnp.mean(jnp.log10(L2s)), path_to_result=path)
    printLog(f'Time to free eq: {t}', path_to_result=path)

    neural_data = {'free_eq': (x, *h), 'path': path, 'complex': {}, 'beta': beta,
                   'grad': true_grads}

    data = {'fdyn': {f'sample {k}': dEds[k,:] for k in range(batch_size)},
            'path': path}
    
    h1 = h
    
    printLog('\ncomplex phase:\n', path_to_result=path)
    cparams = to_complex_dict(params)
    h = to_complex(h1)
    x = x.astype(jnp.complex64)

    data['cdyn'] = {}
    
    for n in range(N):

        dEdz = []
        cbeta = beta * jnp.exp((2 * 1j * jnp.pi * n)/N)
        L2s = model.batched_eval_conv(params, x, y, h, cbeta, True)
        dEdz.append(L2s.mean())
        t, crit = 0, jnp.mean(jnp.log10(L2s)).item()
        printLog(crit, path_to_result=path)
    
        while (t<=T2) and (crit>=-5.8):
            if noise==0.0:
                h, _ = model.batched_fwd(params, x, h, 1, cbeta, y)
            else:
                h, _, key = model.batched_noisy_fwd(params, x, h, 1, cbeta, y, key)
            L2s = model.batched_eval_conv(params, x, y, h, cbeta, True)
            crit = jnp.mean(jnp.log10(L2s))
            t+=1
            dEdz.append(L2s.mean())
    
        printLog(f'\nTime to eq: {t}', path_to_result=path)
        printLog(crit, path_to_result=path)
        neural_data['complex'][str(n)] = (x, *h)
        data['cdyn'][str(n)] = dEdz

    save_data(data, path, 'data')
    save_data(neural_data, path, 'neural_data')



elif args['model_type']=='xent_p_cnn':

    from models import cnn
    
    # Unpacking argument values    
    seed = args['seed']
    task = args['task']
    batch_size = args['batch_size']
    archi = hk.data_structures.to_immutable_dict(args['archi'])
    beta = args['beta']
    T1, T2, N = args['T1'], args['T2'], args['N']
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
        params, _, _, _ = load_checkpoint(params, name, path, n_devices)


    x, y = next(iter(train_loader))
    if n_devices==1:
        h = model.batched_init_neurons(x)  
        y = one_hot(y, n_targets)
    else:
        x, y = split(x), split(y)
        h = model.parallel_init_neurons(x)
        y = par_one_hot(y, n_targets)
    print(y.shape)
 
    dEds = jnp.zeros((batch_size, T1+1))
    L2s = model.batched_eval_conv(params, x, y, h, 0.0, False)
    dEds = dEds.at[:, 0].set(L2s)
    t, crit = 0, jnp.mean(jnp.log10(L2s)).item()
    
    while (t<T1) and (crit>=-5.9):
        h, _ = model.batched_fwd(params, x, h, 1, 0.0, y)
        L2s = model.batched_eval_conv(params, x, y, h, 0.0, False)
        dEds = dEds.at[:, t+1].set(L2s)
        crit = jnp.unique(jnp.mean(jnp.log10(L2s))).item()
        printLog(f'{t}\t{jnp.log10(L2s)}', path_to_result=path)
        t+=1


    neural_data = {'free_eq': (x, *h), 'path': path, 'complex': {}}
    printLog(f'Time to free eq: {t}', path_to_result=path)

    data = {'fdyn': {f'sample {k}': dEds[k,:] for k in range(batch_size)},
            'cdyn': {},
            'path': path}
    


    h1 = h
    
    
    cparams = to_complex_dict(params)
    h = to_complex(h1)
    x = x.astype(jnp.complex64)

    
    for n in range(N):
        h = to_complex(h1)
        
        dEdz = []
        cbeta = beta * jnp.exp((2 * 1j * jnp.pi * n)/N)
        L2s = model.batched_eval_conv(cparams, x, y, h, cbeta, True)
        dEdz.append(L2s.mean())
        t, crit = 0, jnp.mean(jnp.log10(L2s)).item()
    
        while (t<=T2) and (crit>=-5.5):
            h, _ = model.batched_fwd(cparams, x, h, 1, cbeta, y)
            L2s = model.batched_eval_conv(cparams, x, y, h, cbeta, True)
            crit = jnp.mean(jnp.log10(L2s))
            t+=1
            dEdz.append(L2s.mean())
   
        printLog(f'\nTime to eq: {t}', path_to_result=path)
        printLog(jnp.log10(L2s), path_to_result=path)
        neural_data['complex'][str(n)] = (x, *h)
        data['cdyn'][str(n)] = dEdz
 
    save_data(data, path, 'data')
    save_data(neural_data, path, 'neural_data')





