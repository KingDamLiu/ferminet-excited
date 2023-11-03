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

# 定义体系
config_file = 'sample_config/config_minimal.yml'
raw_config, cfg = Configuration.load_configuration_file(config_file)

# cfg.system.electrons = (5,2)  # (alpha electrons, beta electrons)
# cfg.system.molecule = [system.Atom('N', (0, 0, 0))]

symbol, spin = 'Sc', 1
mol = gto.Mole()
# # Set up molecule
mol.build(
    atom=f'{symbol} 0 0 0',
    basis={symbol: 'ccecpccpvdz'},
    ecp={symbol: 'ccecp'},
    spin=int(spin))

cfg.system.pyscf_mol = mol
cfg.system.ecp_quadrature_id = 'icosahedron_12'

# Check if mol is a pyscf molecule and convert to internal representation
if cfg.system.pyscf_mol:
  cfg = pyscf_to_molecule(cfg)

# cfg.system.electrons=(9,6)

# Set training parameters
cfg.batch_size = 4096
cfg.pretrain.n_epochs = 0


writer_manager = None
# 根据设备定义数据
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

# Create parameters, network, and vmaped/pmaped derivations

if cfg.pretrain.method == 'hf' and cfg.pretrain.n_epochs > 0:
  hartree_fock = pretrain.get_hf(
      pyscf_mol=cfg.system.get('pyscf_mol'),
      molecule=cfg.system.molecule,
      nspins=nspins,
      restricted=False,
      basis=cfg.pretrain.basis)
  # broadcast the result of PySCF from host 0 to all other hosts
  hartree_fock.mean_field.mo_coeff = multihost_utils.broadcast_one_to_all(
      hartree_fock.mean_field.mo_coeff
  )

import os

cfg.log.save_path = "experiment_data/ground_states_ecp/Sc_2_2_ecp_ferminet/"
ckpt_restore_filename = "experiment_data/ground_states_ecp/Sc_2_2_ecp_ferminet/ferminet_2023_10_20_05:06:50/qmcjax_ckpt_010000.npz"
cfg.optim.n_epochs = 10000

# ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
# cfg.save(os.path.join(ckpt_save_path, "full_config.yml"))
# files = os.listdir(cfg.log.save_path)
# ckpt_restore_filenames = []
# for file in files:
#     if 'ferminet_2023' in file:
#         ckpt = os.listdir(os.path.join(cfg.log.save_path,file))
#         ckpt.sort()
#     if len(ckpt)>2:
#         ckpt_restore_filenames.append(os.path.join(cfg.log.save_path, file, ckpt[-2]))
# ckpt_restore_filenames.sort()

# 波函数接口
signed_networks = []
sign_networks = []
logabs_networks = []
params = []
datas = []
mcmc_widths = []
pmoves = []
clipping_states = []
mcmc_steps = []

network = create_network(cfg, charges, nspins)
key, subkey = jax.random.split(key)
signed_network = network.apply
logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
sign_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[0]
orbitals = lambda *args, **kwargs: network.orbitals(*args, **kwargs)
batch_network = jax.vmap(
    logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
)
sign_network_vmap = jax.vmap(
    sign_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
)
orbitals_vmap = jax.vmap(
    orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
)
sign_networks.append(sign_network_vmap)
logabs_networks.append(batch_network)
signed_networks.append(signed_network)

# _, _, param, _, _, _ = checkpoint.restore(
#     ckpt_restore_filename, host_batch_size)
# param = kfac_jax.utils.broadcast_all_local_devices(param)
# key, subkey = jax.random.split(key)

param = network.init(subkey)
param = kfac_jax.utils.replicate_all_local_devices(param)
# make sure data on each host is initialized differently
subkey = jax.random.fold_in(subkey, jax.process_index())
pos, spins = init_electrons(
    subkey,
    cfg.system.molecule,
    cfg.system.electrons,
    batch_size=host_batch_size,
    init_width=cfg.mcmc.init_width,
)
pos = jnp.reshape(pos, data_shape + pos.shape[1:])
pos = kfac_jax.utils.broadcast_all_local_devices(pos)
spins = jnp.reshape(spins, data_shape + spins.shape[1:])
spins = kfac_jax.utils.broadcast_all_local_devices(spins)
data = networks.FermiNetData(
    positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
)

clipping_state = init_clipping_state()
clipping_state = kfac_jax.utils.replicate_all_local_devices(clipping_state)

opt_state_ckpt = None
mcmc_width_ckpt = None
# Main training

# 初始化蒙特卡洛采样
if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))
    pmove = np.zeros(cfg.mcmc.adapt_frequency)

# Construct MCMC step
atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
mcmc_step = mcmc.make_mcmc_step(
    logabs_networks[0],
    device_batch_size,
    steps=cfg.mcmc.steps,
    atoms=atoms_to_mcmc,
    blocks=cfg.mcmc.blocks,
)

datas.append(data)
clipping_states.append(clipping_state)
params.append(param)
mcmc_widths.append(mcmc_width)
pmoves.append(pmove)
mcmc_steps.append(mcmc_step)

t_init = 0

sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

# Construct the loss function
local_energy = hamiltonian.local_energy(
f=signed_networks,
charges=charges,
nspins=nspins,
use_scan=False,
complex_output=cfg.network.complex,
pyscf_mole=cfg.system.pyscf_mol,
ecp_quadrature_id=cfg.system.ecp_quadrature_id,
)

clipping_config = constants.ClippingConfig()

evaluate_loss = qmc_loss_functions.make_loss(
logabs_networks,
sign_networks,
local_energy,
clipping_config,
complex_output=cfg.network.complex
)


if t_init == 0:
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_mcmc_step(mcmc_steps=mcmc_steps)

    for t in range(cfg.mcmc.burn_in):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        datas, pmoves_t = burn_in_step(
            datas,
            params,
            key=subkeys,
            mcmc_widths=mcmc_widths)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params,clipping_states, subkeys, datas)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])