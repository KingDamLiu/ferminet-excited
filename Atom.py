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
from ferminet_excited.train import train

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
num_atom = 11

symbol = periodic_table[num_atom-1]
spin = spins[num_atom-1]
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
    cfg = pyscf_to_molecule(cfg)

cfg.log.save_path = "data/expriment_2_1/"+periodic_table[num_atom-1]+file_name
cfg.optim.n_epochs = 20000
cfg.optim.num_psi_updates = 1

# Set training parameters
cfg.batch_size = 2048
cfg.pretrain.n_epochs = 0
