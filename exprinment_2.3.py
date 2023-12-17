# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core training loop for neural QMC in JAX."""

import time
from absl import logging

from ferminet import curvature_tags_and_blocks

from ferminet import networks
from ferminet import pretrain
from ferminet.utils import system
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax

from ferminet_excited.main_utils import *
import numpy as np
import optax

# from ferminet.wavefunction import create_network

import time
import sys
import pandas as pd

from absl import logging
from ferminet.utils import system
from pyscf import gto
import os
os.environ['NVIDIA_TF32_OVERRIDE']="0"

from ferminet_excited import checkpoint
from ferminet_excited import constants
from ferminet_excited import hamiltonian
from ferminet_excited import loss as qmc_loss_functions
from ferminet_excited.loss import init_clipping_state
from ferminet_excited import mcmc
from ferminet_excited import base_config
from ferminet_excited.wavefunction import create_network
from ferminet_excited.configuration import Configuration

# 输出数据
train_schema = ['step', 'E_mean', 'E_mean_clip', 'E_var', 'E_var_clip', 'pmove', 'S','V', 'T', 'V_loc', 'V_nloc', 'delta_time']

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# 读取配置文件
config_file = 'sample_config/config_minimal.yml'
raw_config, cfg = Configuration.load_configuration_file(config_file)

# 初始化设备
num_devices = jax.local_device_count()
num_hosts = jax.device_count() // num_devices
logging.info('Starting QMC with %i XLA devices per host '
              'across %i hosts.', num_devices, num_hosts)
if cfg.batch_size % (num_devices * num_hosts) != 0:
  raise ValueError('Batch size must be divisible by number of devices, '
                    f'got batch size {cfg.batch_size} for '
                    f'{num_devices * num_hosts} devices.')
host_batch_size = cfg.batch_size // num_hosts  # batch size per host
device_batch_size = host_batch_size // num_devices  # batch size per device
data_shape = (num_devices, device_batch_size)

periodic_table = ['H','He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Te', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']

spins = [1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,1,0,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,2,1,0,1,0,1,2,3,2,1]

Num_psi = 3



# 对乙烯进行建模

# 扭转角度
postions = [[-0.675000, 0.000000, 0.000000],
    [0.675000, 0.000000, 0.000000],
    [-1.242900, 0.000000, -0.930370],
    [-1.242900, 0.000000, 0.930370],
    [1.242900, 0.000000, -0.930370],
    [1.242900, 0.000000, 0.930370]]
postions = np.array(postions)
tau = np.array([0,15,30,45,60,70,80,85,90])/180*np.pi
twist = np.array([0.930370*np.sin(tau),0.930370*np.cos(tau)]).T

# # 锥化角度
# postions = [(-0.688500, 0.000000, 0.000000),
#     (0.688500, 0.000000, 0.000000),
#     (-1.307207, 0.000000, -0.915547),
#     (-1.307207, 0.000000, 0.915547),
#     (1.307207, -0.915547, 0.000000),
#     (1.307207, 0.915547, 0.000000)]

# postions = np.array(postions)
# phi = np.array([0,20,40,60,70,80,90,95,97.5,100,102.5,105,110,120])/180*np.pi
# twist = np.array([postions[0,0] + (postions[2,0]-postions[0,0])*np.cos(phi),(postions[2,0]-postions[0,0])*np.sin(phi)]).T
# postions[2,0:2] = twist[1]
# postions[3,0:2] = twist[1]

for twis in range(0, len(twist)):
    postions[2,1:3] = -twist[twis]
    postions[3,1:3] = twist[twis]

    mol = gto.Mole()
    mol.build(
        atom=[['C', postions[0]], ['C', postions[1]], ['H', postions[2]], ['H', postions[3]], ['H', postions[4]], ['H', postions[5]]],
        basis={'C': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
        ecp={'C': 'ccecp', 'H': 'ccecp'},
        spin=0)


    cfg.system.pyscf_mol = mol
    cfg.system.ecp_quadrature_id = 'icosahedron_12'
    file_name = 'ecp'

    # Check if mol is a pyscf molecule and convert to internal representation
    if cfg.system.pyscf_mol:
    # cfg.update(
    #     system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
        cfg = pyscf_to_molecule(cfg)
        # cfg.system.electrons = (3, 1)
    # elif cfg.system.molecule:
    #     cfg = molecule_to_system(cfg)

    cfg.log.save_path = "data/expriment_2_1/C2H4_tau"+str(round(tau[twis]*180/np.pi))+file_name
    cfg.optim.n_epochs = 4000
    cfg.optim.num_psi_updates = 1

    # Set training parameters
    cfg.batch_size = 2048
    cfg.pretrain.n_epochs = 0

    writer_manager = None

    # Convert mol config into array of atomic positions and charges
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    nspins = cfg.system.electrons
    # Generate atomic configurations for each walker
    batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
    batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
    batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
    batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

    if cfg.debug.deterministic:
        seed = 23
    else:
        seed = jnp.asarray([1e6 * time.time()])
        seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
        key = jax.random.PRNGKey(seed)

    # init_path = "data/expriment_2_1/C2H4_tau"+str(round(tau[twis-1]*180/np.pi))+file_name
    # ckpt = os.listdir(init_path)
    # ckpt.sort()
    # init_params = ["data/expriment_2_1/C2H4_tau"+str(round(tau[twis-1]*180/np.pi))+file_name+"/"+ckpt[0]+"/qmcjax_ckpt_004000.npz"]
    init_params = [None]

    # Create parameters, network, and vmaped/pmaped derivations
    for j in range(Num_psi):
        # 待求解波函数参数初始化文件
        ckpt_restore_filenames = []
        ckpt_save_paths = []
        train_stats_file_names = []
        for i in range(cfg.optim.num_psi_updates):
            # 新建待求解波函数地址
            ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
            train_stats_file_name = os.path.join(ckpt_save_path, 'train_stats.csv')
            cfg.save(os.path.join(ckpt_save_path, "full_config.yml"))
            ckpt_restore_filenames.append(init_params[cfg.optim.num_psi_updates-1-i])
            ckpt_save_paths.append(ckpt_save_path)
            train_stats_file_names.append(train_stats_file_name)
            time.sleep(1)
        # 已存在波函数参数文件
        files = os.listdir(cfg.log.save_path)
        files.sort()
        for file in files:
            if 'ferminet_2023' in file:
                ckpt = os.listdir(os.path.join(cfg.log.save_path,file))
                ckpt.sort()
                if len(ckpt)>2:
                    ckpt_restore_filenames.append(os.path.join(cfg.log.save_path, file, ckpt[-2]))
        # ckpt_restore_filenames.sort()
        
        # 初始化波函数及其参数
        signed_networks = []
        sign_networks = []
        logabs_networks = []
        params = []
        datas = []
        mcmc_widths = []
        pmoves = []
        clipping_states = []
        mcmc_steps = []
        for ckpt_restore_filename in ckpt_restore_filenames:

            sign_network_vmap, batch_network, signed_network, param, data, clipping_state, mcmc_width, pmove, mcmc_step = init_wavefunction(
            cfg, atoms, charges, nspins, batch_atoms, batch_charges, key, host_batch_size, device_batch_size, data_shape, ckpt_restore_filename)

            sign_networks.append(sign_network_vmap)
            logabs_networks.append(batch_network)
            signed_networks.append(signed_network)
            datas.append(data)
            clipping_states.append(clipping_state)
            params.append(param)
            mcmc_widths.append(mcmc_width)
            pmoves.append(pmove)
            mcmc_steps.append(mcmc_step)

        t_init = 0
        opt_state_ckpt = None
        sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

        evaluate_loss = make_loss(logabs_networks, signed_networks, sign_networks, charges, nspins, cfg)

        optimizer, opt_state, step = make_optimizer(cfg, evaluate_loss, params, sharded_key, datas, mcmc_steps, clipping_states, opt_state_ckpt=None)


        sharded_key, datas, pmoves_t = init_mcmcs(cfg, evaluate_loss, params, mcmc_steps, datas, clipping_states, mcmc_widths, sharded_key)

        time_of_last_ckpt = time.time()
        weighted_stats = None

        if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
            # If opt_state_ckpt is None, then we're restarting from a previous inference
            # run (most likely due to preemption) and so should continue from the last
            # iteration in the checkpoint. Otherwise, starting an inference run from a
            # training run.
            logging.info('No optimizer provided. Assuming inference run.')
            logging.info('Setting initial iteration to 0.')
            t_init = 0

        logging.info('num_param: %d', get_num_param().get_key(params[0]))
        print('Main training loop')
        start_time = time.time()
        for t in range(t_init, cfg.optim.n_epochs):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            datas, params, opt_state, clipping_states, loss, unused_aux_datas, pmoves_t, params_state = step(
                datas,
                params,
                opt_state,
                subkeys,
                clipping_states,
                mcmc_widths)

            delta_time = time.time() - start_time
            start_time = time.time()
            # due to pmean, loss, and pmove should be the same across
            # devices.
            loss = loss[0]
            for i in range(len(pmoves)):

                pmove = pmoves_t[i][0]

                # Update MCMC move width
                if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
                    if np.mean(pmoves[i]) > 0.55:
                        mcmc_widths[i] *= 1.1
                    if np.mean(pmoves) < 0.5:
                        mcmc_widths[i] /= 1.1
                        pmoves[i][:] = 0
                pmoves[i][t%cfg.mcmc.adapt_frequency] = pmove

            for i, unused_aux_data in zip(range(len(unused_aux_datas)),unused_aux_datas):
                # for unused_aux_data in unused_aux_datas:
                # Logging
                if t % cfg.log.stats_frequency == 0:
                    logging.info(
                        'Step %05d: delta_time=%f, E_mean=%f, E_var=%f, pmove=%f, T=%f, V=%f, V_loc=%f, V_nloc=%f, S=%f', 
                        t,
                        delta_time, 
                        unused_aux_data.E_mean[0], 
                        unused_aux_data.E_var[0], 
                        pmove, 
                        unused_aux_data.T[0], 
                        unused_aux_data.V[0], 
                        unused_aux_data.V_loc[0],
                        unused_aux_data.V_nloc[0],
                        unused_aux_data.S[0].sum())
                out_data = dict(
                    step=t,
                    E_mean=np.asarray(unused_aux_data.E_mean[0]),
                    E_mean_clip=np.asarray(unused_aux_data.E_mean_clipped[0]),
                    E_var_clip=np.asarray(unused_aux_data.E_var_clipped[0]),
                    pmove=np.asarray(pmoves_t[0][0]),
                    E_var=np.asarray(unused_aux_data.E_var[0]),
                    V = np.asarray(unused_aux_data.V[0]),
                    T = np.asarray(unused_aux_data.T[0]),
                    V_loc = np.asarray(unused_aux_data.V_loc[0]),
                    V_nloc = np.asarray(unused_aux_data.V_nloc[0]),
                    S = np.asarray(unused_aux_data.S[0].sum()),
                    f = np.asarray(unused_aux_data.f[0].sum()),
                    delta_time=np.asarray(delta_time),
                    grad_norm = params_state['grad_norm'][0],
                    learning_rate = params_state['learning_rate'][0],
                    momentum = params_state['momentum'][0],
                    param_norm = params_state['param_norm'][0],
                    precon_grad_norm = params_state['precon_grad_norm'][0],
                    update_norm = params_state['update_norm'][0],
                    num_param = get_num_param().get_key(params[0]))
                for si, s in zip(range(unused_aux_data.S[0].shape[0]), unused_aux_data.S[0]):
                    out_data['S_'+str(si)] = np.asarray(s)
                for fi, f in zip(range(unused_aux_data.f[0].shape[0]), unused_aux_data.f[0]):
                    out_data['f_'+str(fi)] = np.asarray(f)
                if t>0 and os.path.exists(train_stats_file_names[i]):
                    df = pd.DataFrame(out_data, index=[0])
                    df.to_csv(train_stats_file_names[i], mode='a', header=False)
                elif t == 0:
                    df = pd.DataFrame(out_data, index=[0])
                    df.to_csv(train_stats_file_names[i], header=True, mode = 'w')

                # Checkpointing
                if (t+1) % cfg.log.save_frequency==0:
                    merge_data = get_from_devices(datas[i])
                    merge_param = get_from_devices(params[i])
                    merge_clipping_state = get_from_devices(clipping_states[i])
                    checkpoint.save(ckpt_save_paths[i], t+1, merge_data, merge_param, opt_state, mcmc_width,merge_clipping_state)
                    time_of_last_ckpt = time.time()
