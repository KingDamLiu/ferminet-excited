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

"""Helper functions to create the loss and custom gradient of the loss."""

from typing import Tuple

import chex
from ferminet_excited import constants
from ferminet_excited import hamiltonian
from ferminet import networks
import jax
import jax.numpy as jnp
import kfac_jax
from typing_extensions import Protocol
from typing import Literal

@chex.dataclass
class AuxiliaryLossData:
  """Auxiliary data returned by total_energy.

  Attributes:
    variance: mean variance over batch, and over all devices if inside a pmap.
    local_energy: local energy for each MCMC configuration.
    clipped_energy: local energy after clipping has been applied
    grad_local_energy: gradient of the local energy.
  """
  E_var:jax.Array
  E_loc:jax.Array
  E_loc_clipped:jax.Array
  E_mean_clipped:jax.Array
  E_var_clipped:jax.Array
  E_mean:jax.Array
  T:jax.Array
  V:jax.Array
  V_nloc:jax.Array
  V_loc:jax.Array
  psi:jax.Array
  grad_local_energy: jax.Array | None
  S: jax.Array | None


class LossFn(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched data elements to pass to the network.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """


def clip_local_values(
    local_values: jnp.ndarray,
    mean_local_values: jnp.ndarray,
    clip_scale: float,
    clip_from_median: bool,
    center_at_clipped_value: bool,
    complex_output: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Clips local operator estimates to remove outliers.

  Args:
    local_values: batch of local values,  Of/f, where f is the wavefunction and
      O is the operator of interest.
    mean_local_values: mean (over the global batch) of the local values.
    clip_scale: clip local quantities that are outside nD of the estimate of the
      expectation value of the operator, where n is this value and D the mean
      absolute deviation of the local quantities from the estimate of w, to the
      boundaries. The clipped local quantities should only be used to evaluate
      gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate/robust to outliers.
    center_at_clipped_value: If true, center the local energy differences passed
      back to the gradient around the clipped quantities, so the mean difference
      across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Tuple of the central value (estimate of the expectation value of the
    operator) and deviations from the central value for each element in the
    batch. If per_device_threshold is True, then the central value is per
    device.
  """

  batch_mean = lambda values: constants.pmean(jnp.mean(values))

  def clip_at_total_variation(values, center, scale):
    tv = batch_mean(jnp.abs(values- center))
    return jnp.clip(values, center - scale * tv, center + scale * tv)

  if clip_from_median:
    # More natural place to center the clipping, but expensive due to both
    # the median and all_gather (at least on multihost)
    clip_center = jnp.median(constants.all_gather(local_values).real)
  else:
    clip_center = mean_local_values
  # roughly, the total variation of the local energies
  if complex_output:
    clipped_local_values = (
        clip_at_total_variation(
            local_values.real, clip_center.real, clip_scale) +
        1.j * clip_at_total_variation(
            local_values.imag, clip_center.imag, clip_scale)
    )
  else:
    clipped_local_values = clip_at_total_variation(
        local_values, clip_center, clip_scale)
  if center_at_clipped_value:
    diff_center = batch_mean(clipped_local_values)
  else:
    diff_center = mean_local_values
  diff = clipped_local_values - diff_center
  return diff_center, diff

class ClippingConfig():
    name: Literal["hard", "tanh"] = "tanh"
    width_metric: Literal["std", "mae"] = "std"
    center: Literal["mean", "median"] = "mean"
    from_previous_step: bool = True
    clip_by: float = 5.0

def init_clipping_state():
    return jnp.array([0.0, 1e5]).squeeze()

def _update_clipping_state(E, clipping_state, clipping_config: ClippingConfig):
    del clipping_state
    center = dict(mean=jnp.nanmean,
                  median=jnp.nanmedian,
                  )[clipping_config.center](E)
    center = constants.pmean(center)
    width = dict(std=jnp.nanstd,
                 mae=lambda x: jnp.nanmean(jnp.abs(x-center)),
                 )[clipping_config.width_metric](E) * clipping_config.clip_by
    width = constants.pmean(width)
    return jnp.array([center, width]).squeeze()

def _clip_energies(E, clipping_state, clipping_config: ClippingConfig):
    if clipping_config.from_previous_step:
        center, width = clipping_state[0], clipping_state[1]
    else:
        clipping_state = _update_clipping_state(E, clipping_state, clipping_config)
        center, width = clipping_state[0], clipping_state[1]

    if clipping_config.name == "hard":
        clipped_energies = jnp.clip(E, center - width, center + width)
    elif clipping_config.name == "tanh":
        clipped_energies = center + jnp.tanh((E - center) / width) * width
    else:
        raise ValueError(f"Unsupported config-value for optimization.clipping.name: {clipping_config.name}")
    new_clipping_state = _update_clipping_state(clipped_energies, clipping_state, clipping_config)
    return clipped_energies, new_clipping_state

def clip_mean(x):
    a = jnp.nanpercentile(x, jnp.array([2, 98]))
    if 1:
        return jnp.nanmean(jnp.clip(x, a[0], a[1]))
    else:
        return jnp.nanmean(x[(x>a[0])&(x<a[1])])

def make_loss(logabs_networks: networks.LogFermiNetLike,
              sign_networks: networks.FermiNetLike,
              local_energy: hamiltonian.LocalEnergy,
              clipping_config,
              complex_output: bool = False,
              num_psi_update: int = 1) -> LossFn:
  """Creates the loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a
      single MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(
      local_energy,
      in_axes=(
          None,
          0,
          0,
      ),
      out_axes=0,
  )

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      clipping_states: jnp.ndarray,
      key: chex.PRNGKey,
      datas: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    clipping_state = clipping_states[0]
    keys = jax.random.split(key, num=datas[0].positions.shape[0])
    E_pots, E_kins, V_locs, V_nlocs = batch_local_energy(params[0:num_psi_update], keys, datas[0:num_psi_update])
    clipping_states_temp = []
    aux_datas = []

    for i, clipping_state, E_pot, E_kin, V_loc, V_nloc, data in zip(range(num_psi_update), clipping_states[0:num_psi_update], E_pots, E_kins, V_locs, V_nlocs, datas[0:num_psi_update]):
      E_loc = E_pot + E_kin + V_nloc
      # 将能量的过大值和过小值剪裁
      E_loc_clipped, clipping_state = _clip_energies(E_loc, clipping_state, clipping_config)
      E_mean_clipped = constants.pmean(jnp.nanmean(E_loc_clipped))
      E_var_clipped = constants.pmean(jnp.nanmean((E_loc_clipped - E_mean_clipped) ** 2))
      T = constants.pmean(jnp.nanmean(E_kin))
      V = constants.pmean(jnp.nanmean(E_pot))
      V_nlocs = constants.pmean(jnp.nanmean(V_nloc))
      V_locs = constants.pmean(jnp.nanmean(V_loc))
      E_mean = constants.pmean(jnp.nanmean(E_loc))
      diff = E_loc - E_mean
      E_var = constants.pmean(jnp.nanmean(diff * jnp.conj(diff)))
      aux_data = AuxiliaryLossData(
          E_var=E_var.real,
          E_loc=E_loc,
          E_loc_clipped=E_loc_clipped,
          E_mean_clipped=E_mean_clipped,
          E_var_clipped=E_var_clipped,
          E_mean = E_mean,
          T=T,
          V=V,
          V_nloc=V_nlocs,
          V_loc=V_locs,
          psi=None,
          S = None,
          grad_local_energy=None,
      )
      # clipping_states_temp.append(clipping_state)
      clipping_states[i] = clipping_state
      aux_datas.append(aux_data)
    
    psi_datas = []
    for i in range(len(params)):
          # 计算波函数各波函数
      psis = []
      for j, data in zip(range(len(datas)), datas):
        primals_in = (params[i], data.positions, data.spins, data.atoms, data.charges)
        psis.append(jnp.exp(logabs_networks[i](*primals_in))*sign_networks[i](*primals_in))
      psi_datas.append(psis)
    aux_datas[0].psi = jnp.array(psi_datas)

    # 计算波函数的重叠损失
    for i in range(num_psi_update):
        S = []
        for j in range(i+1, len(params)):
            psi_ij = constants.pmean(clip_mean(psi_datas[i][j]/psi_datas[j][j]))
            psi_ji = constants.pmean(clip_mean(psi_datas[j][i]/psi_datas[i][i]))
            S.append(jnp.abs(psi_ij*psi_ji)**0.5)
        aux_datas[i].S = jnp.array(S)
        aux_datas[i].psi = [psi_datas[i][j]/psi_datas[j][j],psi_datas[j][i]/psi_datas[i][i]]

    return E_mean, (clipping_states, psi_datas, aux_datas)

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, clipping_states, key, datas = primals
    E_mean, (clipping_states, psi_datas, aux_datas) = total_energy(params, clipping_states, key, datas)

    tangents_outs = 0
    for i in range(num_psi_update):
      diff = aux_datas[i].E_loc_clipped - aux_datas[i].E_mean_clipped
      data = primals[3][i]
      
      data_tangents = tangents[3][i]
      primals_in = (primals[0][i], data.positions, data.spins, data.atoms, data.charges)
      tangents_in = (
          tangents[0][i],
          data_tangents.positions,
          data_tangents.spins,
          data_tangents.atoms,
          data_tangents.charges,
      )
      log_psi_sqr, tangents_log_psi_sqr = jax.jvp(logabs_networks[i], primals_in, tangents_in)
      kfac_jax.register_normal_predictive_distribution(log_psi_sqr[:, None])
      device_batch_size = jnp.shape(aux_datas[i].E_loc_clipped)[0]

      tangents_outs += jnp.dot(tangents_log_psi_sqr, diff) / device_batch_size

      # 波函数重叠的导数
      for j in range(len(params)):
          if i!=j:
              psi_diff = psi_datas[j][i]/psi_datas[i][i]-constants.pmean(clip_mean(psi_datas[j][i]/psi_datas[i][i]))
              # S = jnp.abs(constants.pmean(clip_mean(psi_datas[i][j]/psi_datas[j][j]))*constants.pmean(clip_mean(psi_datas[j][i]/psi_datas[i][i])))**0.5-1e-4
              tangents_outs += jnp.dot(tangents_log_psi_sqr, psi_diff) / device_batch_size * constants.pmean(clip_mean(psi_datas[i][j]/psi_datas[j][j]))# *(1/(1-S)**2)
    primals_out = E_mean, (clipping_states, aux_datas)
    tangents_out = (tangents_outs, (clipping_states, aux_datas))
    return primals_out, tangents_out

  return total_energy

def make_wqmc_loss(
    network: networks.LogFermiNetLike,
    local_energy: hamiltonian.LocalEnergy,
    clip_local_energy: float = 0.0,
    clip_from_median: bool = True,
    center_at_clipped_energy: bool = True,
    complex_output: bool = False,
) -> LossFn:
  """Creates the WQMC loss function, including custom gradients.

  Args:
    network: callable which evaluates the log of the magnitude of the
      wavefunction (square root of the log probability distribution) at a single
      MCMC configuration given the network parameters.
    local_energy: callable which evaluates the local energy.
    clip_local_energy: If greater than zero, clip local energies that are
      outside [E_L - n D, E_L + n D], where E_L is the mean local energy, n is
      this value and D the mean absolute deviation of the local energies from
      the mean, to the boundaries. The clipped local energies are only used to
      evaluate gradients.
    clip_from_median: If true, center the clipping window at the median rather
      than the mean. Potentially expensive in multi-host training, but more
      accurate.
    center_at_clipped_energy: If true, center the local energy differences
      passed back to the gradient around the clipped local energy, so the mean
      difference across the batch is guaranteed to be zero.
    complex_output: If true, the local energies will be complex valued.

  Returns:
    Callable with signature (params, data) and returns (loss, aux_data), where
    loss is the mean energy, and aux_data is an AuxiliaryLossDataobject. The
    loss is averaged over the batch and over all devices inside a pmap.
  """
  batch_local_energy = jax.vmap(
      local_energy,
      in_axes=(
          None,
          0,
          networks.FermiNetData(positions=0, spins=0, atoms=0, charges=0),
      ),
      out_axes=0,
  )
  batch_network = jax.vmap(network, in_axes=(None, 0, 0, 0, 0), out_axes=0)

  @jax.custom_jvp
  def total_energy(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, AuxiliaryLossData]:
    """Evaluates the total energy of the network for a batch of configurations.

    Note: the signature of this function is fixed to match that expected by
    kfac_jax.optimizer.Optimizer with value_func_has_rng=True and
    value_func_has_aux=True.

    Args:
      params: parameters to pass to the network.
      key: PRNG state.
      data: Batched MCMC configurations to pass to the local energy function.

    Returns:
      (loss, aux_data), where loss is the mean energy, and aux_data is an
      AuxiliaryLossData object containing the variance of the energy and the
      local energy per MCMC configuration. The loss and variance are averaged
      over the batch and over all devices inside a pmap.
    """
    keys = jax.random.split(key, num=data.positions.shape[0])
    e_l = batch_local_energy(params, keys, data)
    loss = constants.pmean(jnp.mean(e_l))
    loss_diff = e_l - loss
    variance = constants.pmean(jnp.mean(loss_diff * jnp.conj(loss_diff)))

    def batch_local_energy_pos(pos):
      network_data = networks.FermiNetData(
          positions=pos,
          spins=data.spins,
          atoms=data.atoms,
          charges=data.charges,
      )
      return batch_local_energy(params, keys, network_data).sum()

    grad_e_l = jax.grad(batch_local_energy_pos)(data.positions)
    grad_e_l = jnp.tanh(jax.lax.stop_gradient(grad_e_l))
    return loss, AuxiliaryLossData(
        variance=variance.real,
        local_energy=e_l,
        clipped_energy=e_l,
        grad_local_energy=grad_e_l,
    )

  @total_energy.defjvp
  def total_energy_jvp(primals, tangents):  # pylint: disable=unused-variable
    """Custom Jacobian-vector product for unbiased local energy gradients."""
    params, key, data = primals
    loss, aux_data = total_energy(params, key, data)

    if clip_local_energy > 0.0:
      aux_data.clipped_energy, diff = clip_local_values(
          aux_data.local_energy,
          loss,
          clip_local_energy,
          clip_from_median,
          center_at_clipped_energy,
          complex_output,
      )
    else:
      diff = aux_data.local_energy - loss

    def log_q(params_, pos_, spins_, atoms_, charges_):
      out = batch_network(params_, pos_, spins_, atoms_, charges_)
      kfac_jax.register_normal_predictive_distribution(out[:, None])
      return out.sum()

    score = jax.grad(log_q, argnums=1)
    primals = (params, data.positions, data.spins, data.atoms, data.charges)
    tangents = (
        tangents[0],
        tangents[2].positions,
        tangents[2].spins,
        tangents[2].atoms,
        tangents[2].charges,
    )
    score_primal, score_tangent = jax.jvp(score, primals, tangents)

    score_norm = jnp.linalg.norm(score_primal, axis=-1, keepdims=True)
    median = jnp.median(constants.all_gather(score_norm))
    deviation = jnp.mean(jnp.abs(score_norm - median))
    mask = score_norm < (median + 5 * deviation)
    log_q_tangent_out = (aux_data.grad_local_energy * score_tangent * mask).sum(
        axis=1
    )
    log_q_tangent_out *= len(mask) / mask.sum()

    _, psi_tangent = jax.jvp(batch_network, primals, tangents)
    log_q_tangent_out += diff * psi_tangent
    primals_out = loss, aux_data
    tangents_out = (log_q_tangent_out.mean(), aux_data)
    return primals_out, tangents_out

  return total_energy
