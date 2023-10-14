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

# 输出数据
train_schema = ['step', 'E_mean', 'E_mean_clip', 'E_var', 'E_var_clip', 'pmove', 'S','V', 'T', 'V_loc', 'V_nloc', 'delta_time']

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# 定义体系
cfg = base_config.default()
cfg.system.electrons = (2,2)  # (alpha electrons, beta electrons)
cfg.system.molecule = [system.Atom('Be', (0, 0, 0))]

# symbol, spin = 'Ga', 1
# mol = gto.Mole()
# # # Set up molecule
# mol.build(
#     atom=f'{symbol} 0 0 0',
#     basis={symbol: 'ccecpccpvdz'},
#     ecp={symbol: 'ccecp'},
#     spin=int(spin))

# cfg.system.pyscf_mol = mol
cfg.system.ecp_quadrature_id = 'icosahedron_12'

# Check if mol is a pyscf molecule and convert to internal representation
if cfg.system.pyscf_mol:
  cfg.update(
      system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
  cfg = pyscf_to_molecule(cfg)

# cfg.system.electrons=(9,6)

# Set training parameters
cfg.batch_size = 4096
cfg.pretrain.iterations = 0


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

if cfg.pretrain.method == 'hf' and cfg.pretrain.iterations > 0:
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

cfg.log.save_path = ""

for j in range(16):
  files = os.listdir('.')
  ckpt_restore_filenames = []
  for file in files:
    if 'ferminet_2023' in file:
      ckpt = os.listdir(file)
      ckpt.sort()
      if len(ckpt)>2:
        ckpt_restore_filenames.append(file+'/'+ckpt[-2])
  ckpt_restore_filenames.sort()
  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
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

  init_param = None #'Fe_ecp_ferminet/ferminet_2023_09_25_11:15:15/qmcjax_ckpt_005000.npz'
  # init_param = "K_ecp_ferminet_no_pre/ferminet_2023_09_29_17:47:01/qmcjax_ckpt_010000.npz"
  # if len(ckpt_restore_filenames)>0 and 0:
  #   init_param = ckpt_restore_filenames[-1]
  for i in range(len(ckpt_restore_filenames)+1):
    if i > 0:
      ckpt_restore_filename = ckpt_restore_filenames[i-1]
    else:
      ckpt_restore_filename = None

    network = create_network(cfg, charges, nspins)
    key, subkey = jax.random.split(key)
    signed_network = network.apply
    logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
    sign_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[0]
    batch_network = jax.vmap(
        logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )
    sign_network_vmap = jax.vmap(
        sign_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
    )
    sign_networks.append(sign_network_vmap)
    logabs_networks.append(batch_network)
    signed_networks.append(signed_network)

    if ckpt_restore_filename:
      _, _, param, _, _, _ = checkpoint.restore(
          ckpt_restore_filename, host_batch_size)
      param = kfac_jax.utils.broadcast_all_local_devices(param)
      key, subkey = jax.random.split(key)
    elif init_param:
      _, _, param, _, _, _ = checkpoint.restore(
          init_param, host_batch_size)
      param = kfac_jax.utils.broadcast_all_local_devices(param)
      key, subkey = jax.random.split(key)
    else:
      logging.info('No checkpoint found. Training new model.')
      key, subkey = jax.random.split(key)

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
        logabs_networks[i],
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
      complex_output=cfg.network.get('complex', False),
      pyscf_mole=cfg.system.pyscf_mol,
      ecp_quadrature_id=cfg.system.ecp_quadrature_id,
      )

  clipping_config = constants.ClippingConfig()

  evaluate_loss = qmc_loss_functions.make_loss(
      logabs_networks,
      sign_networks,
      local_energy,
      clipping_config,
      complex_output=cfg.network.get('complex', False)
  )
  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        value_func_has_state=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, datas, clipping_states)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  # Construct training step
  if not optimizer:
    opt_state = None
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_loss_step(evaluate_loss))
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer))
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_steps=mcmc_steps,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')


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

  print('Main training loop')
  train_stats_file_name = os.path.join(ckpt_save_path, 'train_stats.csv')
  start_time = time.time()
  for t in range(t_init, cfg.optim.iterations):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    datas, params, opt_state, clipping_states, loss, unused_aux_datas, pmoves_t = step(
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

    for unused_aux_data in unused_aux_datas:
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
          delta_time=np.asarray(delta_time))
      for si, s in zip(range(unused_aux_data.S[0].shape[0]), unused_aux_data.S[0]):
        out_data['S_'+str(si)] = np.asarray(s)
      if t>0 and os.path.exists(train_stats_file_name):
          df = pd.DataFrame(out_data, index=[0])
          df.to_csv(train_stats_file_name, mode='a', header=False)
      elif t == 0:
          df = pd.DataFrame(out_data, index=[0])
          df.to_csv(train_stats_file_name, header=True, mode = 'w')

    # Checkpointing
    if (t+1) % cfg.log.save_frequency==0:
      checkpoint.save(ckpt_save_path, t+1, datas[0], params[0], opt_state, mcmc_width, clipping_states[0])
      time_of_last_ckpt = time.time()