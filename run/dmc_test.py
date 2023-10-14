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
from ferminet import constants
from ferminet import curvature_tags_and_blocks
from ferminet import envelopes
from ferminet import hamiltonian
from ferminet import loss as qmc_loss_functions
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
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol

import time


def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))


def init_electrons(
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    molecule: system.Atom objects making up the molecule.
    electrons: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """
  if sum(atom.charge for atom in molecule) != sum(electrons):
    if len(molecule) == 1:
      atomic_spin_configs = [electrons]
    else:
      raise NotImplementedError('No initialization policy yet '
                                'exists for charged molecules.')
  else:
    atomic_spin_configs = [
        (atom.element.nalpha, atom.element.nbeta) for atom in molecule
    ]
    assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
    while tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons:
      i = np.random.randint(len(atomic_spin_configs))
      nalpha, nbeta = atomic_spin_configs[i]
      atomic_spin_configs[i] = nbeta, nalpha

  # Assign each electron to an atom initially.
  electron_positions = []
  for i in range(2):
    for j in range(len(molecule)):
      atom_position = jnp.asarray(molecule[j].coords)
      electron_positions.append(
          jnp.tile(atom_position, atomic_spin_configs[j][i]))
  electron_positions = jnp.concatenate(electron_positions)
  # Create a batch of configurations with a Gaussian distribution about each
  # atom.
  key, subkey = jax.random.split(key)
  electron_positions += (
      jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
      * init_width
  )

  electron_spins = _assign_spin_configuration(
      electrons[0], electrons[1], batch_size
  )

  return electron_positions, electron_spins


# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.optimizer.OptimizerState]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]


class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Differentiate wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, state, loss, aux_data = optimizer_step(params, data, state,
                                                       loss_key)
    return data, new_params, state, loss, aux_data, pmove

  return step


def make_kfac_training_step(mcmc_step, damping: float,
                            optimizer: kfac_jax.Optimizer) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))

  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.optimizer.OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    # Optimization step
    new_params, state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
    )
    return data, new_params, state, stats['loss'], stats['aux'], pmove

  return step

import sys

from absl import logging
from ferminet.utils import system
from ferminet import base_config

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
cfg = base_config.default()
cfg.system.electrons = (15,11)  # (alpha electrons, beta electrons)
cfg.system.molecule = [system.Atom('Fe', (0, 0, 0))]

# Set training parameters
cfg.batch_size = 2048
cfg.pretrain.iterations = 0

# train(cfg)
"""Runs training loop for QMC.

Args:
  cfg: ConfigDict containing the system and training parameters to run on. See
    base_config.default for more details.
  writer_manager: context manager with a write method for logging output. If
    None, a default writer (ferminet.utils.writers.Writer) is used.

Raises:
  ValueError: if an illegal or unsupported value in cfg is detected.
"""
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

# Check if mol is a pyscf molecule and convert to internal representation
if cfg.system.pyscf_mol:
  cfg.update(
      system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))

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

if cfg.network.make_feature_layer_fn:
  feature_layer_module, feature_layer_fn = (
      cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
  feature_layer_module = importlib.import_module(feature_layer_module)
  make_feature_layer: networks.MakeFeatureLayer = getattr(
      feature_layer_module, feature_layer_fn
  )
  feature_layer = make_feature_layer(
      natoms=charges.shape[0],
      nspins=cfg.system.electrons,
      ndim=cfg.system.ndim,
      **cfg.network.make_feature_layer_kwargs)
else:
  feature_layer = networks.make_ferminet_features(
      natoms=charges.shape[0],
      nspins=cfg.system.electrons,
      ndim=cfg.system.ndim,
      rescale_inputs=cfg.network.get('rescale_inputs', False),
  )

if cfg.network.make_envelope_fn:
  envelope_module, envelope_fn = (
      cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
  envelope_module = importlib.import_module(envelope_module)
  make_envelope = getattr(envelope_module, envelope_fn)
  envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
else:
  envelope = envelopes.make_isotropic_envelope()

if cfg.network.network_type == 'ferminet':
  network = networks.make_fermi_net(
      nspins,
      charges,
      ndim=cfg.system.ndim,
      determinants=cfg.network.determinants,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=cfg.network.get('jastrow', 'default'),
      bias_orbitals=cfg.network.bias_orbitals,
      full_det=cfg.network.full_det,
      rescale_inputs=cfg.network.get('rescale_inputs', False),
      complex_output=cfg.network.get('complex', False),
      **cfg.network.ferminet,
  )
elif cfg.network.network_type == 'psiformer':
  network = psiformer.make_fermi_net(
      nspins,
      charges,
      ndim=cfg.system.ndim,
      determinants=cfg.network.determinants,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=cfg.network.get('jastrow', 'default'),
      bias_orbitals=cfg.network.bias_orbitals,
      rescale_inputs=cfg.network.get('rescale_inputs', False),
      complex_output=cfg.network.get('complex', False),
      **cfg.network.psiformer,
  )
key, subkey = jax.random.split(key)

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


ckpt_restore_filename = 'ferminet_2023_09_06_07:48:58/qmcjax_ckpt_002661.npz'

t_init, data, params, opt_state_ckpt, mcmc_width_ckpt = checkpoint.restore(
    ckpt_restore_filename, host_batch_size)

position = data.positions.reshape((-1, data.positions.shape[-1]))

# Get a single copy of network params from the replicated one
single_params = jax.tree_map(lambda x: x[0], params)
# batch_network = jax.vmap(signed_network, in_axes=(None, 0, 0, 0, 0), out_axes=0)
network_wrapper = lambda x: signed_network(single_params, x, data.spins[0,0], data.atoms[0,0], data.charges[0,0])

from dmc_config import get_config
from jaqmc.dmc import run
from jaqmc.dmc.ckpt_metric_manager import DataPath

dmc_cfg = get_config()
dmc_cfg.log.save_path = 'dmc_ckpt'
dmc_cfg.log.remote_save_path = 'dmc_ckpt'

run(
    data.positions[0],
    dmc_cfg.iterations,
    network_wrapper,
    dmc_cfg.time_step, key,
    nuclei=atoms,
    charges=charges,

    # Below are optional arguments
    mixed_estimator_num_steps=dmc_cfg.mixed_estimator_num_steps,
    energy_window_size=dmc_cfg.energy_window_size,
    weight_branch_threshold=dmc_cfg.weight_branch_threshold,
    update_energy_offset_interval=dmc_cfg.update_energy_offset_interval,
    energy_offset_update_amplitude=dmc_cfg.energy_offset_update_amplitude,
    energy_cutoff_alpha=dmc_cfg.energy_cutoff_alpha,
    effective_time_step_update_period=dmc_cfg.effective_time_step_update_period,
    energy_outlier_rel_threshold=dmc_cfg.energy_outlier_rel_threshold,
    fix_size=dmc_cfg.fix_size,
    ebye_move=dmc_cfg.ebye_move,
    block_size=dmc_cfg.block_size,
    max_restore_nums=dmc_cfg.max_restore_nums,
    save_path=DataPath(dmc_cfg.log.save_path, dmc_cfg.log.remote_save_path),
)