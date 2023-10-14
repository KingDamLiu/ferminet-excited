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

import functools
import importlib
import time
from typing import Optional, Sequence, Tuple, Union

from absl import logging
import chex
from ferminet import checkpoint
from ferminet_clipping import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet_clipping import hamiltonian
from ferminet_clipping import loss as qmc_loss_functions
from ferminet import mcmc
from ferminet import networks
from ferminet import pretrain
from ferminet import psiformer
from ferminet.utils import statistics
from ferminet.utils import system
from ferminet.utils import writers
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax

from ferminet_clipping.main_utils import *
import numpy as np
import optax

from ferminet_clipping.wavefunction import create_network
from ferminet_clipping.loss import init_clipping_state

import time
import sys

from absl import logging
from ferminet.utils import system
from ferminet_clipping import base_config
from pyscf import gto

# Set up logging
train_schema = ['step', 'E_mean', 'E_mean_clip', 'E_var', 'E_var_clip', 'pmove','V', 'T', 'V_loc', 'V_nloc', 'delta_time']

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
# cfg.system.electrons = (10,6)  # (alpha electrons, beta electrons)
# cfg.system.molecule = [system.Atom('Fe', (0, 0, 0))]

symbol, spin = 'Fe', 4
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
  cfg.update(
      system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
  cfg = pyscf_to_molecule(cfg)

# cfg.system.electrons=(9,6)

# Set training parameters
cfg.batch_size = 2048
cfg.pretrain.iterations = 0


writer_manager = None
# Device logging
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
clipping_state = init_clipping_state()
# Generate atomic configurations for each walker
batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)
clipping_state = kfac_jax.utils.replicate_all_local_devices(clipping_state)

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

network = create_network(cfg, charges, nspins)
key, subkey = jax.random.split(key)

# params0 = network.init(subkey)
# params0 = kfac_jax.utils.replicate_all_local_devices(params0)
# params1 = network.init(subkey)
# params1 = kfac_jax.utils.replicate_all_local_devices(params1)
# params = [params0,params1]

params = network.init(subkey)
params = kfac_jax.utils.replicate_all_local_devices(params)

signed_network = network.apply
# Often just need log|psi(x)|.
sign_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[0]
logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
batch_network = jax.vmap(
    logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
)  # batched network

# Exclusively when computing the gradient wrt the energy for complex
# wavefunctions, it is necessary to have log(psi) rather than log(|psi|).
# This is unused if the wavefunction is real-valued.
def log_network(*args, **kwargs):
  if not cfg.network.get('complex', False):
    raise ValueError('This function should never be used if the '
                      'wavefunction is real-valued.')
  phase, mag = signed_network(*args, **kwargs)
  return mag + 1.j * phase

# Set up checkpointing and restore params/data if necessary
# Mirror behaviour of checkpoints in TF FermiNet.
# Checkpoints are saved to save_path.
# When restoring, we first check for a checkpoint in save_path. If none are
# found, then we check in restore_path.  This enables calculations to be
# started from a previous calculation but then resume from their own
# checkpoints in the event of pre-emption.

ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

ckpt_restore_filename = (
    checkpoint.find_last_checkpoint(ckpt_save_path) or
    checkpoint.find_last_checkpoint(ckpt_restore_path))

if ckpt_restore_filename:
  t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
      ckpt_restore_filename, host_batch_size)
else:
  logging.info('No checkpoint found. Training new model.')
  key, subkey = jax.random.split(key)
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

  t_init = 0
  opt_state_ckpt = None
  mcmc_width_ckpt = None

# Initialisation done. We now want to have different PRNG streams on each
# device. Shard the key over devices
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

# Pretraining to match Hartree-Fock

if (
    t_init == 0
    and cfg.pretrain.method == 'hf'
    and cfg.pretrain.iterations > 0
):
  pretrain_spins = spins[0, 0]
  batch_orbitals = jax.vmap(
      network.orbitals, in_axes=(None, 0, 0, 0, 0), out_axes=0
  )
  sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
  params, data.positions = pretrain.pretrain_hartree_fock(
      params=params,
      positions=data.positions,
      spins=pretrain_spins,
      atoms=data.atoms,
      charges=data.charges,
      batch_network=batch_network,
      batch_orbitals=batch_orbitals,
      network_options=network.options,
      sharded_key=subkeys,
      electrons=cfg.system.electrons,
      scf_approx=hartree_fock,
      iterations=cfg.pretrain.iterations,
  )

# Main training

# Construct MCMC step
atoms_to_mcmc = atoms if cfg.mcmc.scale_by_nuclear_distance else None
mcmc_step = mcmc.make_mcmc_step(
    batch_network,
    device_batch_size,
    steps=cfg.mcmc.steps,
    atoms=atoms_to_mcmc,
    blocks=cfg.mcmc.blocks,
)
# Construct loss and optimizer
if cfg.system.make_local_energy_fn:
  local_energy_module, local_energy_fn = (
      cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
  local_energy_module = importlib.import_module(local_energy_module)
  make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
  local_energy = make_local_energy(
      f=signed_network,
      charges=charges,
      nspins=nspins,
      use_scan=False,
      **cfg.system.make_local_energy_kwargs)
else:
  local_energy = hamiltonian.local_energy(
      f=signed_network,
      charges=charges,
      nspins=nspins,
      use_scan=False,
      complex_output=cfg.network.get('complex', False),
      pyscf_mole=cfg.system.pyscf_mol,
      ecp_quadrature_id=cfg.system.ecp_quadrature_id,
      )

clipping_config = constants.ClippingConfig()

evaluate_loss = qmc_loss_functions.make_loss(
    log_network if cfg.network.get('complex', False) else logabs_network,
    sign_network,
    local_energy,
    clipping_config,
    # clip_local_energy=cfg.optim.clip_local_energy,
    # clip_from_median=cfg.optim.clip_median,
    # center_at_clipped_energy=cfg.optim.center_at_clip,
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
  opt_state = optimizer.init([params], subkeys, [data], [clipping_state])
  opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
else:
  raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

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
      mcmc_step=mcmc_step,
      damping=cfg.optim.kfac.damping,
      optimizer=optimizer)
else:
  raise ValueError(f'Unknown optimizer: {optimizer}')

if mcmc_width_ckpt is not None:
  mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
else:
  mcmc_width = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(cfg.mcmc.move_width))
pmoves = np.zeros(cfg.mcmc.adapt_frequency)

if t_init == 0:
  logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

  burn_in_step = make_training_step(
      mcmc_step=mcmc_step, optimizer_step=null_update)

  for t in range(cfg.mcmc.burn_in):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    data, params, *_ = burn_in_step(
        data,
        params,
        state=None,
        key=subkeys,
        mcmc_width=mcmc_width)
  logging.info('Completed burn-in MCMC steps')
  sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
  ptotal_energy = constants.pmap(evaluate_loss)
  initial_energy, _ = ptotal_energy([params],[clipping_state], subkeys, [data])
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

params = [params]
datas = [data]
clipping_states = [clipping_state]

if writer_manager is None:
  writer_manager = writers.Writer(
      name='train_stats',
      schema=train_schema,
      directory=ckpt_save_path,
      iteration_key=None,
      log=False)
with writer_manager as writer:
  # Main training loop
  start_time = time.time()
  for t in range(t_init, cfg.optim.iterations):
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    datas, params, opt_state, clipping_states, loss, unused_aux_data, pmove = step(
        datas,
        params,
        opt_state,
        subkeys,
        clipping_states,
        mcmc_width)
    delta_time = time.time() - start_time
    start_time = time.time()
    # due to pmean, loss, and pmove should be the same across
    # devices.
    loss = loss[0]
    # per batch variance isn't informative. Use weighted mean and variance
    # instead.
    # weighted_stats = statistics.exponentialy_weighted_stats(
    #     alpha=0.1, observation=loss, previous_stats=weighted_stats)
    pmove = pmove[0]

    # Update MCMC move width
    if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
      if np.mean(pmoves) > 0.55:
        mcmc_width *= 1.1
      if np.mean(pmoves) < 0.5:
        mcmc_width /= 1.1
      pmoves[:] = 0
    pmoves[t%cfg.mcmc.adapt_frequency] = pmove

    if cfg.debug.check_nan:
      tree = {'params': params, 'loss': loss}
      if cfg.optim.optimizer != 'none':
        tree['optim'] = opt_state
      chex.assert_tree_all_finite(tree)

    # Logging
    if t % cfg.log.stats_frequency == 0:
      logging.info(
          'Step %05d: delta_time=%f, E_mean=%f, E_var=%f, pmove=%f, T=%f, V=%f, V_loc=%f, V_nloc=%f', 
          t,
          delta_time, 
          unused_aux_data.E_mean[0], 
          unused_aux_data.E_var[0], 
          pmove, 
          unused_aux_data.T[0], 
          unused_aux_data.V[0], 
          unused_aux_data.V_loc[0], 
          unused_aux_data.V_nloc[0])
      writer.write(
          t,
          step=t,
          E_mean=np.asarray(unused_aux_data.E_mean[0]),
          E_mean_clip=np.asarray(unused_aux_data.E_mean_clipped[0]),
          E_var_clip=np.asarray(unused_aux_data.E_var_clipped[0]),
          # ewmean=np.asarray(weighted_stats.mean),
          # ewvar=np.asarray(weighted_stats.variance),
          pmove=np.asarray(pmove),
          E_var=np.asarray(unused_aux_data.E_var[0]),
          V = np.asarray(unused_aux_data.V[0]),
          T = np.asarray(unused_aux_data.T[0]),
          V_loc = np.asarray(unused_aux_data.V_loc[0]),
          V_nloc = np.asarray(unused_aux_data.V_nloc[0]),
          delta_time=np.asarray(delta_time))

    # Checkpointing
    if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60:
      checkpoint.save(ckpt_save_path, t, datas[0], params[0], opt_state, mcmc_width)
      time_of_last_ckpt = time.time()