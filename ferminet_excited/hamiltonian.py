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

"""Evaluating the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Sequence, Union

import chex
from ferminet import networks
import jax
from jax import lax
import folx
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol

from ferminet_excited.integral import pseudoPotential
from ferminet_excited.integral.quadrature import get_quadrature


Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> jnp.ndarray:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f: networks.FermiNetLike,
      charges: jnp.ndarray,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


KineticEnergy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]


def select_output(f: Callable[..., Sequence[Any]],
                  argnum: int) -> Callable[..., Any]:
  """Return the argnum-th result from callable f."""

  def f_selected(*args, **kwargs):
    return f(*args, **kwargs)[argnum]

  return f_selected


def local_kinetic_energy(
    f: networks.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = select_output(f, 0)
  logabs_f = select_output(f, 1)

  def _lapl_over_f(params, data):
    n = data.positions.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(logabs_f, argnums=1)
    def grad_f_closure(x):
      return grad_f(params, x, data.spins, data.atoms, data.charges)

    primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

    if complex_output:
      grad_phase = jax.grad(phase_f, argnums=1)
      def grad_phase_closure(x):
        return grad_phase(params, x, data.spins, data.atoms, data.charges)
      phase_primal, dgrad_phase = jax.linearize(
          grad_phase_closure, data.positions)
      hessian_diagonal = (
          lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i]
      )
    else:
      hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

    if use_scan:
      _, diagonal = lax.scan(
          lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
      result = -0.5 * jnp.sum(diagonal)
    else:
      result = -0.5 * lax.fori_loop(
          0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
    result -= 0.5 * jnp.sum(primal ** 2)
    if complex_output:
      result += 0.5 * jnp.sum(phase_primal ** 2)
      result -= 1.j * jnp.sum(primal * phase_primal)
    return result
  
  def _lapl_over_f_folx(params, data):
    f_closure = lambda x: logabs_f(params,
                                    x,
                                    data.spins,
                                    data.atoms,
                                    data.charges)
    f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
    output = f_wrapped(data.positions)
    return - (output.laplacian +
              jnp.sum(output.jacobian.dense_array ** 2)) / 2

  def _grad_over_f(params, data):
    grad_f = jax.grad(logabs_f, argnums=1)
    def grad_f_closure(x):
      return grad_f(params, x, data.spins, data.atoms, data.charges)

    primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

    result = 0.5 * jnp.sum(primal ** 2)
    return result

  def _lapl_over_f_optimized(params, data):
      n = data.positions.shape[0]
      eye = jnp.eye(n)
      
      # 封装 logabs_f 以包含所有参数
      def logabs_f_closure(x):
          return logabs_f(params, x, data.spins, data.atoms, data.charges)
      
      # 封装 phase_f 以包含所有参数，如果有复数输出
      if complex_output:
          def phase_f_closure(x):
              return phase_f(params, x, data.spins, data.atoms, data.charges)

      # 定义一个函数来计算每个对角元素
      def hessian_diagonal_element(x, idx):
          _, vector_jacobian_prod = jax.jvp(logabs_f_closure, (x,), (eye[idx],))
          return vector_jacobian_prod[idx]

      # 对所有维度并行化对角 Hessian 的计算
      diagonal_elements = jax.vmap(hessian_diagonal_element, in_axes=(None, 0))(data.positions, jnp.arange(n))

      # 如果处理复数输出
      if complex_output:
          def hessian_diagonal_element_phase(x, idx):
              _, vector_jacobian_prod = jax.jvp(phase_f_closure, (x,), (eye[idx],))
              return vector_jacobian_prod[idx]
          
          phase_diagonal_elements = jax.vmap(hessian_diagonal_element_phase, in_axes=(None, 0))(data.positions, jnp.arange(n))
          diagonal_elements = diagonal_elements + 1.j * phase_diagonal_elements

      # 计算最终结果
      result = -0.5 * jnp.sum(diagonal_elements)
      primal = logabs_f_closure(data.positions)
      result -= 0.5 * jnp.sum(primal ** 2)

      if complex_output:
          phase_primal = phase_f_closure(data.positions)
          result += 0.5 * jnp.sum(phase_primal ** 2)
          result -= 1.j * jnp.sum(primal * phase_primal)

      return result
  
  def _lapl_over_f_c(params, data):
      def psi(x):
          return jnp.exp(logabs_f(params, x, data.spins, data.atoms, data.charges))*phase_f(params, x, data.spins, data.atoms, data.charges)
      input_size = data.positions.shape[-1]
      h = jnp.eye(input_size)*1e-4
      print(data.positions.shape)
      result = 0
      for i in range(20):
        result += psi(data.positions+h[i:i+1,:])+ psi(data.positions-h[i:i+1,:])
      return -(result-2*input_size*psi(data.positions))/1e-8/2
  return _lapl_over_f_folx


def potential_electron_electron(r_ee: Array) -> jnp.ndarray:
  """Returns the electron-electron potential.

  Args:
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
  """
  r_ee = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
  return (1.0 / r_ee).sum()


def potential_electron_nuclear(charges: Array, r_ae: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def potential_energy(r_ae: Array, r_ee: Array, atoms: Array,
                     charges: Array) -> jnp.ndarray:
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  return (potential_electron_electron(r_ee) +
          potential_electron_nuclear(charges, r_ae) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    pyscf_mole = None,
    ecp_quadrature_id = None,
) -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  del nspins
  kes = []
  for func in f:
    kes.append(local_kinetic_energy(func,
                              use_scan=use_scan,
                              complex_output=complex_output))

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, datas: networks.FermiNetData
  ) -> jnp.ndarray:
    """Returns the total energy.

    Args:
      params: network parameters.
      key: RNG state.
      data: MCMC configuration.
    """
    del key  # unused
    E_pots = []
    E_kins = []
    V_locs = []
    V_nlocs = []
    for func, param, data,ke in zip(f, params, datas, kes):
      _, _, r_ae, r_ee = networks.construct_input_features(
          data.positions, data.atoms
      )
      potential = potential_energy(r_ae, r_ee, data.atoms, data.charges)
      kinetic = ke(param, data)

      V_loc = 0
      V_nloc = 0
      if pyscf_mole and pyscf_mole._ecp:
        el_ecp = non_local_energy(func, pyscf_mole,
                                        ecp_quadrature_id=ecp_quadrature_id)
        V_loc, V_nloc = el_ecp(param, data)
      E_pots.append(potential)
      E_kins.append(kinetic)
      V_locs.append(V_loc)
      V_nlocs.append(V_nloc)
    return E_pots, E_kins, V_locs, V_nlocs

  return _e_l


def ecp(pe, pa, ecp_coe):
    """
    read ecp coeffs from pyscf obj

    NEWLY ADDED
    """
    norm = jnp.linalg.norm(pe[:, None, :] - pa, axis=-1)
    res = []
    for _, l in ecp_coe:
        result = 0
        for power, coe in enumerate(l):
            for coeff in coe:
                result = result + norm[:, 0] ** (power - 2) * jnp.exp(- coeff[0] * norm[:, 0] ** 2) * \
                         coeff[1]
        res.append(result)
    res = jnp.stack(res, axis=-1)
    return res

def non_local_energy(fs, pyscf_mole, ecp_quadrature_id=None):
    """
    Calculate Ecp energy.

    NEWLY ADDED
    """
    quadrature = get_quadrature(ecp_quadrature_id)

    def psi(params, positions, spins, atoms, charges):

        sign_and_log = fs(params, positions, spins, atoms, charges)

        return jnp.exp(sign_and_log[1]) * sign_and_log[0]

    def non_local(pe, pa, psi, l_list):
        res = pseudoPotential.numerical_integral_exact(psi, pa, pe, l_list, quadrature)
        return res / (4 * jnp.pi * psi(pe))

    def non_local_sum(params, x):
        V_local = 0
        V_nloc = 0
        pe = x.positions.reshape(-1, 3)
        for sym, coord in pyscf_mole._atom:
            v_loc = 0
            v_nloc = 0
            if sym in pyscf_mole._ecp:
                pa = jnp.array(coord)
                ecp_coe = pyscf_mole._ecp[sym][1]
                l_list = list(range(len(ecp_coe) - 1))
                ecp_list = ecp(pe, pa, ecp_coe)
                v_nloc = (jnp.sum(ecp_list[..., 1:] * non_local(pe, pa, lambda p: psi(params, p,  x.spins, x.atoms, x.charges), l_list), axis=-1) + ecp_list[..., 0])
                v_loc = (ecp_list[..., 0])
            V_local = V_local + v_loc
            V_nloc = V_nloc + v_nloc

        return jnp.sum(V_local, axis=-1), jnp.sum(V_nloc, axis=-1)

    return non_local_sum

def non_local_energy_temp(fs, pyscf_mole, ecp_quadrature_id=None):
    """
    Calculate Ecp energy.

    NEWLY ADDED
    """
    quadrature = get_quadrature(ecp_quadrature_id)

    def psi(params, positions, spins, atoms, charges):

        sign_and_log = fs(params, positions, spins, atoms, charges)

        return jnp.exp(sign_and_log[1]) * sign_and_log[0]

    def non_local(pe, pa, psi, l_list):
        res = pseudoPotential.numerical_integral_exact(psi, pa, pe, l_list, quadrature)
        return res / (4 * jnp.pi * psi(pe))

    def non_local_sum(params, x):
        res = 0

        pe = x.positions.reshape(-1, 3)
        for sym, coord in pyscf_mole._atom:
            result = 0

            if sym in pyscf_mole._ecp:
                pa = jnp.array(coord)
                ecp_coe = pyscf_mole._ecp[sym][1]
                l_list = list(range(len(ecp_coe) - 1))
                ecp_list = ecp(pe, pa, ecp_coe)
                result = (jnp.sum(ecp_list[..., 1:] * non_local(pe, pa, lambda p: psi(params, p,  x.spins, x.atoms, x.charges), l_list), axis=-1)+ecp_list[..., 0])

            res = res + result
        return jnp.sum(res, axis=-1)

    return non_local_sum
