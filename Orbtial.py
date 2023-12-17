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

Num_psi = 12


symbol = "Ga"
spin = 1
mol = gto.Mole()
# # Set up molecule
mol.build(
    atom=f'{symbol} 0 0 0',
    basis={symbol: 'ccecpccpvdz'},
    ecp={symbol: 'ccecp'},
    spin=int(spin))

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

cfg.log.save_path = "post_process/Li"+file_name
cfg.optim.n_epochs = 200
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

init_params = ['data/expriment_2_1/Gaecp/ferminet_2023_11_03_23:36:25/qmcjax_ckpt_020000.npz']

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
    orbital_networks = []
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


    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_mcmc_step(mcmc_steps=mcmc_steps)

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
        datas, pmoves_t = burn_in_step(
            datas,
            params,
            key=subkeys,
            mcmc_widths=mcmc_widths)
        logging.info('Completed burn-in MCMC steps')
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        ptotal_energy = constants.pmap(evaluate_loss)
        initial_energy, datas_temp = ptotal_energy(params,clipping_states, subkeys, datas)
        clipping_states, psi_datas, aux_datas = datas_temp
        logging.info('Initial energy: %03.4f E_h', initial_energy[0])
        time_of_last_ckpt = time.time()