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
Num_psi = 1
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

r1 = jnp.linspace(0.1, 2.1,10)
r2 = 1.1
r3 = jnp.linspace(0.1, 2.1,10)
r4 = 1.1

phi1 = 0
phi2 = 0
phi3 = jnp.pi/2
phi4 = 3*jnp.pi/2

theta1 = jnp.linspace(-1,1,10)
theta2 = jnp.pi/2
theta3 = jnp.pi/2
theta4 = jnp.pi/2

R1,R3,Theta1 = jnp.meshgrid(r1,r3,theta1)

X1 = R1*jnp.sin(Theta1)*jnp.cos(phi1)
Y1 = R1*jnp.sin(Theta1)*jnp.sin(phi1)
Z1 = R1*jnp.cos(Theta1)

X2 = r2*jnp.sin(theta2)*jnp.cos(phi2)
Y2 = r2*jnp.sin(theta2)*jnp.sin(phi2)
Z2 = r2*jnp.cos(theta2)

X3 = R3*jnp.sin(theta3)*jnp.cos(phi3)
Y3 = R3*jnp.sin(theta3)*jnp.sin(phi3)
Z3 = R3*jnp.cos(theta3)

X4 = r4*jnp.sin(theta4)*jnp.cos(phi4)
Y4 = r4*jnp.sin(theta4)*jnp.sin(phi4)
Z4 = r4*jnp.cos(theta4)

x = jnp.linspace(-5,5,30)
X,Y,Z = jnp.meshgrid(x,x,x)
position = jnp.array([X.reshape(-1,1), Y.reshape(-1,1), Z.reshape(-1,1)]).transpose(2,1,0)
device_batch_size = position.shape[1]

positions_up = jnp.concatenate([position[0], jnp.tile(jnp.ones(nspins[0]*3-3), reps=(device_batch_size, 1))], 1)[None]
if nspins[1]-1>0:
    positions_down = jnp.concatenate([position[0], jnp.tile(jnp.ones(nspins[1]*3-3), reps=(device_batch_size, 1))])
else:
    positions_down = position
positions = jnp.concatenate([positions_up,positions_down],2)

atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
charges = jnp.array([atom.charge for atom in cfg.system.molecule])
nspins = cfg.system.electrons
# Generate atomic configurations for each walker
batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

spins = jnp.concatenate((jnp.ones(nspins[0]), -jnp.ones(nspins[1])))
spins = jnp.tile(spins[None], reps=(device_batch_size, 1))

# 电子坐标
datas[0].positions = kfac_jax.utils.broadcast_all_local_devices(positions)
# 电子自旋
datas[0].spins = kfac_jax.utils.broadcast_all_local_devices(spins[None])
# 原子坐标
datas[0].atoms = batch_atoms #kfac_jax.utils.broadcast_all_local_devices(batch_atoms)
# 核电荷
datas[0].charges = batch_charges #kfac_jax.utils.broadcast_all_local_devices(charges)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
ptotal_energy = constants.pmap(evaluate_loss)
initial_energy, datas_temp = ptotal_energy(params,clipping_states, subkeys, datas)
clipping_states, psi_datas, aux_datas = datas_temp

spin_id = 0
det_id = 0
orb_id = 1
e_id = 0

S = aux_datas[0].orbital[0][0][spin_id][0,:,det_id, e_id, orb_id].reshape(30,30,30)
# S_min = S.min()
# S_max = S.max()
# S = (S-S_min)/(S_max-S_min)

import plotly.graph_objects as go
import numpy as np
values = np.sin(X*Y*Z) / (X*Y*Z)
fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=S.flatten(),
    # isomin=0.1,
    # isomax=0.8,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    colorscale ='picnic',
    ))
fig.show()