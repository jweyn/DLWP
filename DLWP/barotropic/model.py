"""Non-divergent barotropic vorticity equation dynamical core."""
# (c) Copyright 2016 Andrew Dawson.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import (absolute_import, division, print_function)  #noqa

from datetime import timedelta
import math

import numpy as np

from .pyspharm_transforms import TransformsEngine


class BarotropicModel(object):
    """
    Dynamical core for a spectral non-divergent barotropic vorticity
    equation model.
    The model solves the non-divergent barotropic vorticity equation
    using a spectral method by representing the vorticity as sum of
    spherical harmonics.
    """

    def __init__(self, z, truncation, dt, start_time,
                 robert_coefficient=0.04, damping_coefficient=1e-4,
                 damping_order=4):
        """
        Initialize a barotropic model.
        Arguments:
        * z : numpy.ndarray[nlat, nlon]
            An initial field of height on a global regular grid. In general
            nlon is double nlat.
        * truncation : int
            The spectral truncation (triangular). A suggested value is
            nlon // 3.
        * dt : float
            The model time-step in seconds.
        * start_time : datetime.datetime
            A datetime object representing the start time of the model
            run. This doesn't affect computation, it is only used for
            metadata.
        Optional arguments:
        * robert_coefficient : default 0.04
            The coefficient for the Robert time filter.
        * damping coefficient : default 1e-4
            The coefficient for the damping term.
        * damping_order : default 4 (hyperdiffusion)
            The order of the damping.
        """
        # Model grid size:
        self.nlat, self.nlon = z.shape
        # Filtering properties:
        self.robert_coefficient = robert_coefficient
        # Initialize the spectral transforms engine:
        self.truncation = truncation
        self.engine = TransformsEngine(self.nlon, self.nlat, truncation)
        # Initialize constants for spectral damping:
        m, n = self.engine.wavenumbers
        el = (m + n) * (m + n + 1) / float(self.engine.radius) ** 2
        self.damping = damping_coefficient * (el / el[truncation]) ** damping_order
        # Initialize the grid and spectral model variables:
        self.z_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)  # @jweyn
        self.u_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)
        self.v_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)
        self.vrt_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)
        nspec = (truncation + 1) * (truncation + 2) // 2
        self.vrt_spec = np.zeros([nspec], dtype=np.complex128)
        self.vrt_spec_prev = np.zeros([nspec], dtype=np.complex128)
        # Set the initial state:
        self.set_state(z)  # @jweyn
        # Pre-compute the Coriolis parameter on the model grid:
        lats, _ = self.engine.grid_latlon
        self.f = 2 * 7.29e-5 * np.sin(np.deg2rad(lats))[:, np.newaxis]
        # Set time control parameters:
        self.start_time = start_time
        self.t = 0
        self.dt = dt
        self.first_step = True

    @property
    def valid_time(self):
        """
        A datetime.datetime object representing the current valid time
        of the model state.
        """
        return self.start_time + timedelta(seconds=self.t)

    def set_state(self, z):
        """
        Set the model state from an initial z. @jweyn
        The vorticity must be in grid space, this method will transform
        the vorticity to spectral space using the model truncation, and
        the initial grid vorticity will be computed by converting the
        spectral vorticity back to grid space. This ensures the initial
        grid and spectral vorticity fields are equivalent.
        Argument:
        * vrt : numpy.ndarray[nlat, nlon]
            The model grid vorticity.
        """
        self.z_grid[:] = z  # @jweyn
        vrt = self.engine.grid_to_spec(self.get_vrt(z))  # @jweyn
        # Convert grid vorticity to spectral vorticity:  # @jweyn
        self.vrt_spec[:] = vrt  # @jweyn
        # Compute grid vorticity from spectral vorticity (to ensure it is
        # consistent with the spectral form):
        self.vrt_grid[:] = self.engine.spec_to_grid(self.vrt_spec)
        # Compute the wind components from the spectral vorticity, assuming
        # no divergence:
        self.u_grid[:], self.v_grid[:] = self.engine.uv_grid_from_vrtdiv_spec(
            self.vrt_spec, np.zeros_like(self.vrt_spec))
        # Set the spectral vorticity at the previous time to the current time,
        # which makes sure damping works properly:
        self.vrt_spec_prev[:] = self.vrt_spec

    def step_forward(self):
        """Step the model forward in time by one time-step."""
        if self.first_step:
            dt = self.dt
        else:
            dt = 2 * self.dt
        dudt = -(self.f + self.vrt_grid) * self.v_grid  # @jweyn
        dvdt = (self.f + self.vrt_grid) * self.u_grid  # @jweyn
        dzetadt, _ = self.engine.vrtdiv_spec_from_uv_grid(dudt, dvdt)
        coeffs = 1. / (1. + self.damping * self.dt)
        dzetadt = coeffs * (dzetadt - self.damping * self.vrt_spec_prev)
        if self.first_step:
            # Apply a forward-difference time integration scheme:
            new_vrt_spec = self.vrt_spec + dt * dzetadt
            self.vrt_spec[:] += (self.robert_coefficient *
                                 (new_vrt_spec - self.vrt_spec))
            # Only do the first step once:
            self.first_step = False
        else:
            # Apply a leapfrog time integration scheme:
            self.vrt_spec[:] += (self.robert_coefficient *
                                 (self.vrt_spec_prev - 2. * self.vrt_spec))
            new_vrt_spec = self.vrt_spec_prev + dt * dzetadt
            self.vrt_spec[:] += self.robert_coefficient * new_vrt_spec
        # Overwrite the t-1 time with the current time:
        self.vrt_spec_prev[:] = self.vrt_spec
        # Update the current time with the new values:
        self.vrt_spec[:] = new_vrt_spec
        self.vrt_grid[:] = self.engine.spec_to_grid(new_vrt_spec)
        self.z_grid[:] = self.get_z(self.vrt_grid)  # @jweyn
        self.u_grid[:], self.v_grid[:] = self.engine.uv_grid_from_vrtdiv_spec(
            new_vrt_spec, np.zeros_like(new_vrt_spec))
        # Increment the model time:
        self.t += self.dt

    def run_with_snapshots(self, run_time, snapshot_start=0,
                           snapshot_interval=None):
        """
        A generator that runs the model for a specific amount of time,
        yielding at specified intervals.
        Argument:
        * run_time : float
            The amount of time to run for in seconds.
        Keyword arguments:
        * snapshot_start : default 0
            Don't yield until at least this amount of time has passed,
            measured in seconds.
        * snapshot_interval : float
            The interval between snapshots in seconds.
        """
        snapshot_interval = snapshot_interval or self.dt
        if snapshot_interval < self.dt:
            snapshot_interval = self.dt
        target_steps = int(math.ceil((self.t + run_time) / self.dt))
        step_interval = int(math.ceil(snapshot_interval / self.dt))
        start_time = self.t
        n = 0
        while n <= target_steps:
            self.step_forward()
            n += 1
            if self.t > snapshot_start and n % step_interval == 0:
                yield self.t

    def get_z(self, vrt):  # @jweyn
        n = self.engine.wavenumbers[1] + 1.
        factor = -1 * n * (n + 1) / (self.engine.radius ** 2.)
        z_spec = self.engine.grid_to_spec(vrt) / factor
        return self.engine.spec_to_grid(z_spec)

    def get_vrt(self, z):  # @jweyn
        n = self.engine.wavenumbers[1] + 1.
        factor = -1 * n * (n + 1) / (self.engine.radius ** 2.)
        vrt_spec = factor * self.engine.grid_to_spec(z)
        return self.engine.spec_to_grid(vrt_spec)


class BarotropicModelPsi(object):
    """
    Dynamical core for a spectral non-divergent barotropic vorticity
    equation model. Uses the streamfunction formulation.
    """

    def __init__(self, z, truncation, dt, start_time,
                 robert_coefficient=0.04, damping_coefficient=1e-4,
                 damping_order=4):
        """
        Initialize a barotropic model.
        Arguments:
        * z : numpy.ndarray[nlat, nlon]
            An initial field of geopotential height on a global regular grid.
            In general nlon is double nlat.
        * truncation : int
            The spectral truncation (triangular). A suggested value is
            nlon // 3.
        * dt : float
            The model time-step in seconds.
        * start_time : datetime.datetime
            A datetime object representing the start time of the model
            run. This doesn't affect computation, it is only used for
            metadata.
        Optional arguments:
        * robert_coefficient : default 0.04
            The coefficient for the Robert time filter.
        * damping coefficient : default 1e-4
            The coefficient for the damping term.
        * damping_order : default 4 (hyperdiffusion)
            The order of the damping.
        """
        # Model grid size:
        self.nlat, self.nlon = z.shape
        # Filtering properties:
        self.robert_coefficient = robert_coefficient
        # Initialize the spectral transforms engine:
        self.truncation = truncation
        self.engine = TransformsEngine(self.nlon, self.nlat, truncation)
        # Initialize constants for spectral damping:
        m, n = self.engine.wavenumbers
        el = (m + n) * (m + n + 1) / float(self.engine.radius) ** 2
        self.damping = damping_coefficient * (el / el[truncation]) ** damping_order
        # Initialize the grid variables:
        self.z_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)
        self.psi_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)
        self.vrt_grid = np.zeros([self.nlat, self.nlon], dtype=np.float64)
        # Initialize the spectral variables
        nspec = (truncation + 1) * (truncation + 2) // 2
        self.vrt_spec = np.zeros([nspec], dtype=np.complex128)
        self.vrt_spec_prev = np.zeros([nspec], dtype=np.complex128)
        # Pre-compute the Coriolis parameter on the model grid:
        lats, _ = self.engine.grid_latlon
        self.lats = lats
        self.f = 2 * 7.29e-5
        self.beta = 2 * 7.29e-5 * np.cos(np.deg2rad(lats))[:, np.newaxis] / self.engine.radius
        self.g = 9.81
        # Set the initial state:
        self._set_state(z)
        # Set time control parameters:
        self.start_time = start_time
        self.t = 0
        self.dt = dt
        self.first_step = True

    @property
    def valid_time(self):
        """
        A datetime.datetime object representing the current valid time
        of the model state.
        """
        return self.start_time + timedelta(seconds=self.t)

    def _set_state(self, z):
        """
        Set the model state from an initial z.
        Argument:
        * z : numpy.ndarray[nlat, nlon]
            The model grid geopotential height.
        """
        self.z_grid[:] = z  # @jweyn
        self.psi_grid[:] = self.g * z / self.f
        self.vrt_spec[:] = self._psi_to_vrt(self.engine.grid_to_spec(self.psi_grid))
        # Compute grid vorticity from spectral vorticity (to ensure it is
        # consistent with the spectral form):
        self.vrt_grid[:] = self.engine.spec_to_grid(self.vrt_spec)
        # Set the spectral vorticity at the previous time to the current time,
        # which makes sure damping works properly:
        self.vrt_spec_prev[:] = self.vrt_spec

    def step_forward(self, correct_sh=True):
        """Step the model forward in time by one time-step."""
        psi_spec = self.engine.grid_to_spec(self.psi_grid)
        # dpsidx, _ = self.engine.grad_of_spec(psi_spec)
        # beta_term = self.engine.grid_to_spec(self.beta * dpsidx)
        dzetadt = -1. * (self._J(psi_spec, self.vrt_spec))
        if correct_sh:
            dzetadt = self.engine.spec_to_grid(dzetadt)
            dzetadt[self.lats < 0] = -1. * dzetadt[self.lats < 0]
            dzetadt = self.engine.grid_to_spec(dzetadt)

        coeffs = 1. / (1. + self.damping * self.dt)
        dzetadt = coeffs * (dzetadt - self.damping * self.vrt_spec_prev)

        if self.first_step:
            # Apply a forward-difference time integration scheme:
            dt = self.dt
            new_vrt_spec = self.vrt_spec + dt * dzetadt
            self.vrt_spec[:] += (self.robert_coefficient *
                                 (new_vrt_spec - self.vrt_spec))
            # Only do the first step once:
            self.first_step = False
        else:
            # Apply a leapfrog time integration scheme:
            dt = 2 * self.dt
            self.vrt_spec[:] += (self.robert_coefficient *
                                 (self.vrt_spec_prev - 2. * self.vrt_spec))
            new_vrt_spec = self.vrt_spec_prev + dt * dzetadt
            self.vrt_spec[:] += self.robert_coefficient * new_vrt_spec

        # Update in time
        # Overwrite the t-1 time with the current time:
        self.vrt_spec_prev[:] = self.vrt_spec
        # Update the current time with the new values:
        self.vrt_spec[:] = new_vrt_spec
        self.vrt_grid[:] = self.engine.spec_to_grid(new_vrt_spec)
        self.psi_grid[:] = self.engine.spec_to_grid(self._vrt_to_psi(new_vrt_spec))
        self.z_grid[:] = self.f * self.psi_grid / self.g
        # Increment the model time:
        self.t += self.dt

    def _vrt_to_psi(self, vrt):  # @jweyn
        n = self.engine.wavenumbers[1] + 1.
        factor = -1 * n * (n + 1) / (self.engine.radius ** 2.)
        return vrt / factor

    def _psi_to_vrt(self, z):  # @jweyn
        n = self.engine.wavenumbers[1] + 1.
        factor = -1 * n * (n + 1) / (self.engine.radius ** 2.)
        return factor * z

    def _J(self, psi, vrt):
        dpsidx, dpsidy = self.engine.grad_of_spec(psi)
        dvrtdx, dvrtdy = self.engine.grad_of_spec(vrt)
        return self.engine.grid_to_spec(dpsidx * dvrtdy - dpsidy * dvrtdx)
