from typing import Optional, Sequence, Tuple, Union
import ml_collections
import jax.numpy as jnp
import jax
import numpy as np
from ferminet.utils import system
import optax
from ferminet import networks
from typing_extensions import Protocol
import kfac_jax
from ferminet_excited import loss as qmc_loss_functions
import chex
from ferminet_excited import constants
import functools
import copy

def get_from_devices(data):
    return jax.tree_util.tree_map(lambda x: x[0], data)

class get_num_param:
    size = 0
    def get_key(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                self.get_key(data[key])
        if not isinstance(data, (dict, list, int)):
            self.size += data.size
        if isinstance(data, (tuple, list)):
            self.size += len(data)
            for i in range(len(data)):
                self.get_key(data[i])
        return self.size

def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))

def pyscf_to_molecule(cfg: ml_collections.ConfigDict):
    """Converts the PySCF 'Molecule' in the config to the internal representation.

    Args:
      cfg: ConfigDict containing the system and training parameters to run on. See
        base_config.default for more details. Must have the system.pyscf_mol set.

    Returns:
      cfg: ConfigDict matching the input with system.molecule, system.electrons
        and pretrain.basis fields set from the information in the system.pyscf_mol
        field.

    Raises:
      ValueError: if the system.pyscf_mol field is not set in the cfg.

    MODIFICATION FROM FERMINET: added ECP config
    """
    if not cfg.system.pyscf_mol:
        raise ValueError('You must set system.pyscf_mol in your cfg')
    cfg.system.pyscf_mol.build()
    cfg.system.electrons = cfg.system.pyscf_mol.nelec
    cfg.system.molecule = [system.Atom(cfg.system.pyscf_mol.atom_symbol(i),
                                       cfg.system.pyscf_mol.atom_coords()[i],
                                       charge=cfg.system.pyscf_mol.atom_charges()[i], )
                           for i in range(cfg.system.pyscf_mol.natm)]
    ##  cfg.system.pyscf_mol.atom_charges()[i] return the screen charge of i atom if ecp is used

    cfg.pretrain.basis = str(cfg.system.pyscf_mol.basis)
    cfg.system.ecp = str(cfg.system.pyscf_mol.ecp)
    return cfg

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
        (atom.element.nalpha - int((atom.atomic_number - atom.charge) // 2),
          atom.element.nbeta - int((atom.atomic_number - atom.charge) // 2))
        for atom in molecule
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

def make_mcmc_step(mcmc_steps):
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
  mcmc_steps_temp = []
  for mcmc_step in mcmc_steps:
    mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    mcmc_steps_temp.append(mcmc_step)

  def step(
      datas: networks.FermiNetData,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      mcmc_widths: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    datas_temp = []
    pmoves_temp = []
    for i in range(len(datas)):
      data = copy.deepcopy(datas[i])
      mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
      data, pmove = mcmc_steps_temp[i](params[i], data, mcmc_keys, mcmc_widths[i])
      datas_temp.append(data)
      pmoves_temp.append(pmove)

    return datas_temp, pmoves_temp

  return step


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


def make_kfac_training_step(mcmc_steps, damping: float,
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
  mcmc_steps_temp = []
  for mcmc_step in mcmc_steps:
    mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
    mcmc_steps_temp.append(mcmc_step)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))

  def step(
      datas: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.optimizer.OptimizerState,
      key: chex.PRNGKey,
      clipping_state,
      mcmc_widths: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    datas_temp = []
    pmoves_temp = []
    for i in range(len(datas)):
      data = copy.deepcopy(datas[i])
      mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
      data, pmove = mcmc_steps_temp[i](params[i], data, mcmc_keys, mcmc_widths[i])
      datas_temp.append(data)
      pmoves_temp.append(pmove)

    # Optimization step
    new_params, state, clipping_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=datas_temp,
        momentum=shared_mom,
        damping=shared_damping,
        func_state=clipping_state,
    )
    params_state = dict(
      grad_norm = stats['grad_norm'],
      learning_rate = stats['learning_rate'],
      momentum = stats['momentum'],
      param_norm = stats['param_norm'],
      precon_grad_norm = stats['precon_grad_norm'],
      update_norm = stats['update_norm'],
    )
    return datas_temp, new_params, state, clipping_state, stats['loss'], stats['aux'], pmoves_temp, params_state

  return step