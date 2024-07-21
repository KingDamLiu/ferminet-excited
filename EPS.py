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

LiH_mol = gto.Mole()
LiH_mol.build(
    atom=f'Li 0 0 0; H 1.595 0 0',
    basis={'Li': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
    ecp={'Li': 'ccecp'},
    spin=0)

CO_mol = gto.Mole()
CO_mol.build(
    atom=f'C 0 0 -0.661165; O 0 0 0.472379',
    basis={'C': 'ccecpccpvdz', 'O': 'ccecpccpvdz'},
    ecp={'C': 'ccecp', 'O': 'ccecp'},
    spin=0)

BeH_mol = gto.Mole()
BeH_mol.build(
    atom=f'Be 0 0 0; H 1.326903 0 0',
    basis={'Be': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
    ecp={'Be': 'ccecp'},
    spin=1)

C6H6_mol = gto.Mole()
C6H6_mol.build(
    atom=f'C 0.000000, 1.396792, 0.000000; C 0.000000, -1.396792, 0.000000; C 1.209657, 0.698396, 0.000000; C -1.209657, -0.698396, 0.000000; C -1.209657, 0.698396, 0.000000; C 1.209657, -0.698396, 0.000000; H 0.000000, 2.484212, 0.000000; H 2.151390, 1.242106, 0.000000;H -2.151390, -1.242106, 0.000000;H -2.151390, 1.242106, 0.000000;H 2.151390, -1.242106, 0.000000; H 0.000000, -2.484212, 0.000000',
    basis={'C': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
    ecp={'C': 'ccecp'},
    spin=0)

H2O_mol = gto.Mole()
H2O_mol.build(
    atom=f'O 0.000000, 0.000000, -0.069903; H 0.000000, 0.757532, 0.518435; H 0.000000, -0.757532, 0.518435',
    basis={'O': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
    ecp={'O': 'ccecp'},
    spin=0)

H2S_mol = gto.Mole()
H2S_mol.build(
    atom=f'S 0.       ,  0.       , -0.26652056; H 0.       ,  0.96219289,  0.66259489; H 0.       , -0.96219289,  0.66259489',
    basis={'S': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
    ecp={'S': 'ccecp'},
    spin=0)

HCl_mol = gto.Mole()
HCl_mol.build(
    atom=f'H 0.000000, 0.000000, 0.000000; Cl 0.000000, 0.000000, 1.27517379',
    basis={'H': 'ccecpccpvdz', 'Cl': 'ccecpccpvdz'},
    spin=0)

H2CSi_mol = gto.Mole()
H2CSi_mol.build(
     atom=f'C 0.00000000 0.00000000 0.60851638; Si 0.00000000 0.00000000 -1.10883755; H 0.00000000 0.90452009 -1.70868401; H 0.00000000 -0.90452009 -1.70868401',
     basis={'Si': 'ccecpccpvdz', 'C': 'ccecpccpvdz', 'H': 'ccecpccpvdz'},
     ecp={'Si': 'ccecp', 'C': 'ccecp'},
     spin = 0)


sys = ["LiH_ccecp", "BeH_ccecp", "CO_ccecp", "H2O_ccecp", "H2S_ccecp", "H2CSi_ccecp"]
mols = [LiH_mol, BeH_mol, CO_mol, H2O_mol, H2S_mol, H2CSi_mol]

id = 5

cfg.system.pyscf_mol = mols[id]
cfg.system.ecp_quadrature_id = 'icosahedron_12'
file_name = 'ecp'

# Check if mol is a pyscf molecule and convert to internal representation
if cfg.system.pyscf_mol:
    cfg = pyscf_to_molecule(cfg)

cfg.log.save_path = "/home/gengzi/deepwavefunction/Excited_Calculate/Exp/Moleculer/"+sys[id]
cfg.optim.n_epochs = 20000
# cfg.optim.lr.rate = 1e-5
cfg.optim.num_psi_updates = 1

# Set training parameters
cfg.batch_size = 1024
cfg.pretrain.n_epochs = 0
init_params = [None]

train(cfg, Num_psi=12, init_params=init_params)

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

for twis in range(4, 5):
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

    cfg.log.save_path = "data/expriment_2_1/C2H4/C2H4_tau"+str(round(tau[twis]*180/np.pi))+file_name
    cfg.optim.n_epochs = 4000
    cfg.optim.num_psi_updates = 1

    # Set training parameters
    cfg.batch_size = 2048
    cfg.pretrain.n_epochs = 0