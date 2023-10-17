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
from ferminet_excited.wavefunction import create_network
from absl import logging
from ferminet_excited import mcmc
from ferminet_excited import checkpoint
from ferminet_excited import hamiltonian
from ferminet import curvature_tags_and_blocks

def get_from_devices(data):
    return jax.tree_util.tree_map(lambda x: x[0], data)

def init_clipping_state():
    return jnp.array([0.0, 1e5]).squeeze()

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

def init_mcmcs(cfg, evaluate_loss, params, mcmc_steps, datas, clipping_states, mcmc_widths, sharded_key):
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

    return sharded_key, datas, pmoves_t

def make_optimizer(cfg, evaluate_loss, params, sharded_key, datas, mcmc_steps, clipping_states, opt_state_ckpt=None):
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
        include_norms_in_stats = True,
        # include_per_param_norms_in_stats = True,
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
        mcmc_step=mcmc_steps,
        optimizer_step=make_loss_step(evaluate_loss))
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(
        mcmc_step=mcmc_steps,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer))
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_steps=mcmc_steps,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')
  
  return optimizer, opt_state, step

def make_loss(logabs_networks, signed_networks, sign_networks, charges, nspins, cfg):
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
      complex_output=cfg.network.complex,
      num_psi_update=cfg.optim.num_psi_updates,
  )
  return evaluate_loss

def init_wavefunction(cfg, atoms, charges, nspins, batch_atoms, batch_charges, key, host_batch_size, device_batch_size, data_shape, ckpt_restore_filename=None):
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

    if ckpt_restore_filename:
      _, _, param, _, _, _ = checkpoint.restore(
          ckpt_restore_filename, host_batch_size)
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
        batch_network,
        device_batch_size,
        steps=cfg.mcmc.steps,
        atoms=atoms_to_mcmc,
        blocks=cfg.mcmc.blocks,
    )

    return sign_network_vmap, batch_network, signed_network, param, data, clipping_state, mcmc_width, pmove, mcmc_step

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