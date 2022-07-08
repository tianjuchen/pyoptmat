# pylint: disable=unused-import
"""
  Objects and helper functions to help with deterministic model calibration
  and statistical inference.

  These include the key classes

  - :py:class:`pyoptmat.optimize.DeterministicModel`
  - :py:class:`pyoptmat.optimize.StatisticalModel`
  - :py:class:`pyoptmat.optimize.HierarchicalStatisticalModel`

  which implement

  - deterministic model prediction and optimization
  - single-level statistical model prediction and inference
  - hierarchical statistical model prediction and inference

  These three classes share some common features.  One input to all three is the
  "model maker" function.  This is a function that takes as input as :code:`*args` a set of
  parameters (which can be :py:mod:`torch` tensors, :py:mod:`torch` Parameters, or
  :py:mod`:`pyro` samples as appropriate) and returns a valid :py:class:`torch.nn.Module`.
  Calling this module in turn provides the integrated model predictions.
  The maker function can also take :code:`**kwargs`, for example to provide values that
  should not be optimized or hyperparameters describing how to integrate the model
  results.

  The classes also take a :code:`list` of :code:`str` names, one for each
  parameter the :code:`maker` function takes as input.  These names are used
  to describe the various :py:mod:`torch` :code:`Parameters` or :py:mod:`pyro`
  :code:`samples` and :code:`parameters`.  The specific details of how these
  are used depends on the type of model.

  Parameter values, for example as starting locations or descriptions of the priors,
  are given as :code:`lists` of :py:class:`torch.tensor` objects, again one
  for each model parameter.
"""

import torch
from torch.nn import Module, Parameter
from torch import nn
import torch.nn.functional as F
import pyro
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro import poutine
from pyoptmat import experiments
from pyro.contrib.autoguide import AutoDelta, init_to_mean, AutoMultivariateNormal
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormalizingFlow
from pyro.distributions.transforms import block_autoregressive, iterated
from functools import partial
import contextlib


def bound_factor(mean, factor, min_bound=None):
    """
    Apply the bounded_scale_function map but set the upper bound as mean*(1+factor)
    and the lower bound as mean*(1-factor)

    Args:
      mean (torch.tensor):          center value
      factor (torch.tensor):        bounds factor

    Keyword Args:
      min_bound (torch.tensor):     clip to avoid going lower than this value
    """
    return bounded_scale_function(
        (mean * (1.0 - factor), mean * (1 + factor)), min_bound=min_bound
    )


def bounded_scale_function(bounds, min_bound=None):
    """
    Sets up a scaling function that maps `(0,1)` to `(bounds[0], bounds[1])`
    and clips the values to remain in that range

    Args:
      bounds (tuple(torch.tensor,torch.tensor)):    tuple giving the parameter bounds

    Additional Args:
      min_bounds (torch.tensor):                    clip to avoid going lower than this value
    """
    if min_bound is None:
        return lambda x: torch.clamp(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]
    return lambda x: torch.maximum(
        torch.clamp(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0], min_bound
    )


def clamp_scale_function(bounds):
    """
    Just clamp to bounds

    Args:
      bounds (tuple(torch.tensor, torch.tensor)):   tuple giving the parameter bounds
    """
    return lambda x: torch.clamp(x, bounds[0], bounds[1])


def bound_and_scale(scale, bounds):
    """
    Combination of scaling and bounding

    Args:
      scale (torch.tensor):     divide input by this value
      bounds (torch.tensor):    tuple, clamp to (bounds[0], bounds[1])
    """
    return lambda x: torch.clamp(x / scale, bounds[0], bounds[1])


def log_bound_and_scale(scale, bounds):
    """
    Scale, de-log, and bound

    Args:
      scale (torch.tensor):     divide input by this value, then take exp
      bounds (torch.tensor):    tuple, clamp to (bounds[0], bounds[1])
    """
    return lambda x: torch.clamp(torch.exp(x / scale), bounds[0], bounds[1])


class DeterministicModel(Module):
    """
    Wrap a material model to provide a :py:mod:`pytorch` deterministic model

    Args:
      maker (function):         function that returns a valid model as a
                                :py:class:`pytorch.nn.Module`,
                                given the input parameters
      names (list(str)):        names to use for the parameters
      ics (list(torch.tensor)): initial conditions to use for each parameter
    """

    def __init__(self, maker, names, ics):
        super().__init__()

        self.maker = maker

        self.names = names

        # Add all the parameters
        self.params = names
        for name, ic in zip(names, ics):
            setattr(self, name, Parameter(ic))

    def get_params(self):
        """
        Return the parameters for input to the model
        """
        return [getattr(self, name) for name in self.params]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control):
        """
        Integrate forward and return the results.

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag
        """
        model = self.maker(*self.get_params())

        predictions = model.solve_both(
            exp_data[0], exp_data[1], exp_data[2], exp_control
        )

        return experiments.convert_results(predictions[:, :, 0], exp_cycles, exp_types)


class StatisticalModel(PyroModule):
    """
    Wrap a material model to provide a py:mod:`pyro` single-level
    statistical model

    Single level means each parameter is sampled once before running
    all the tests -- i.e. each test is run on the same material properties.

    Generally this is not appropriate for fitting models (see
    :py:class:`pyoptmat.optimize.HierarchicalStatisticsModel`)
    but can be a nice way to evaluate models (i.e. run all tests on a
    single "heat" of material multiple times to sample heat-to-heat variation).

    Args:
      maker (function):             function that returns a valid
                                    :py:class:`torch.nn.Module`, given the input
                                    parameters
      names (list(str)):            names to use for the parameters
      loc (list(torch.tensor)):     parameter location priors
      scales (list(torch.tensor):   parameter scale priors
      eps (list or scalar):         random noise, can be either a single scalar
                                    or a 1D tensor if it's a 1D tensor then each
                                    entry i represents the noise in test type i
    """

    def __init__(self, maker, names, locs, scales, eps):
        super().__init__()

        self.maker = maker

        # Add all the parameters
        self.params = names
        for name, loc, scale in zip(names, locs, scales):
            setattr(self, name, PyroSample(dist.Normal(loc, scale)))

        self.eps = eps

        self.type_noise = self.eps.dim() > 0

    def get_params(self):
        """
        Return the sampled parameters for input to the model
        """
        return [getattr(self, name) for name in self.params]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
        """
        Integrate forward and return the result

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        model = self.maker(*self.get_params())
        predictions = model.solve_both(
            exp_data[0], exp_data[1], exp_data[2], exp_control
        )
        results = experiments.convert_results(
            predictions[:, :, 0], exp_cycles, exp_types
        )

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1])
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = self.eps[i]
        else:
            full_noise = self.eps

        with pyro.plate("trials", exp_data.shape[2]):
            with pyro.plate("time", exp_data.shape[1]):
                return pyro.sample(
                    "obs", dist.Normal(results, full_noise), obs=exp_results
                )


class HierarchicalStatisticalModel(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise

        self.names = names

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.Normal(loc_loc, loc_scale).to_event(dim)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.HalfNormal(scale_scale).to_event(dim)),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroSample(
                    lambda self, var=var, dim=loc_loc.dim(): dist.Normal(
                        getattr(self, var + loc_suffix),
                        getattr(self, var + scale_suffix),
                    ).to_event(dim)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
            else:
                self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        self.extra_param_names = []

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
            # Setup and sample the top-level loc and scale
            top_loc_samples = []
            top_scale_samples = []
            for var, loc_loc, loc_scale, scale_scale, in zip(
                self.names,
                self.loc_loc_priors,
                self.loc_scale_priors,
                self.scale_scale_priors,
            ):
                dim = loc_loc.dim()
                loc_param = pyro.param(
                    var + self.loc_suffix + self.param_suffix,
                    loc_loc,
                    constraint=constraints.interval(0.0, 1.0),
                )
                scale_param = pyro.param(
                    var + self.scale_suffix + self.param_suffix,
                    scale_scale,
                    constraint=constraints.positive,
                )

                top_loc_samples.append(
                    pyro.sample(
                        var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)
                    )
                )
                top_scale_samples.append(
                    pyro.sample(
                        var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)
                    )
                )

            # Add in the noise, if included in the inference
            if self.include_noise:
                eps_param = pyro.param(
                    "eps" + self.param_suffix,
                    torch.tensor(self.noise_prior),
                    constraint=constraints.positive,
                )
                if self.type_noise:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
                else:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param))

            # Plate on experiments and sample individual values
            with pyro.plate("trials", exp_data.shape[2]):
                for (name, val, dim) in zip(self.names, self.loc_loc_priors, self.dims):
                    # Fix this to init to the mean (or a sample I guess)
                    ll_param = pyro.param(
                        name + self.param_suffix,
                        torch.zeros_like(val)
                        .unsqueeze(0)
                        .repeat((exp_data.shape[2],) + (1,) * dim)
                        + 0.5,
                        constraint=constraints.interval(0.0, 1.0),
                    )
                    param_value = pyro.sample(name, dist.Delta(ll_param).to_event(dim))

        self.extra_param_names = [var + self.param_suffix for var in self.names]

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with pyro.plate("trials", exp_data.shape[2]):
            # Sample the bottom level parameters
            bmodel = self.maker(
                *self.sample_bot(), extra_params=self.get_extra_params()
            )
            # Generate the results
            predictions = bmodel.solve_both(
                exp_data[0], exp_data[1], exp_data[2], exp_control
            )
            # Process the results
            results = experiments.convert_results(
                predictions[:, :, 0], exp_cycles, exp_types
            )

            # Sample!
            with pyro.plate("time", exp_data.shape[1]):
                pyro.sample("obs", dist.Normal(results, full_noise), obs=exp_results)

        return results


class HierarchicalWeightsModel(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        new_weights,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise
        self.new_weights = new_weights
        self.names = names

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.Normal(loc_loc, loc_scale).to_event(dim)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.HalfNormal(scale_scale).to_event(dim)),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroSample(
                    lambda self, var=var, dim=loc_loc.dim(): dist.Normal(
                        getattr(self, var + loc_suffix),
                        getattr(self, var + scale_suffix),
                    ).to_event(dim)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
            else:
                self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        self.extra_param_names = []

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
            # Setup and sample the top-level loc and scale
            top_loc_samples = []
            top_scale_samples = []
            for var, loc_loc, loc_scale, scale_scale, in zip(
                self.names,
                self.loc_loc_priors,
                self.loc_scale_priors,
                self.scale_scale_priors,
            ):
                dim = loc_loc.dim()
                loc_param = pyro.param(
                    var + self.loc_suffix + self.param_suffix,
                    loc_loc,
                    constraint=constraints.interval(0.0, 1.0),
                )
                scale_param = pyro.param(
                    var + self.scale_suffix + self.param_suffix,
                    scale_scale,
                    constraint=constraints.positive,
                )

                top_loc_samples.append(
                    pyro.sample(
                        var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)
                    )
                )
                top_scale_samples.append(
                    pyro.sample(
                        var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)
                    )
                )

            # Add in the noise, if included in the inference
            if self.include_noise:
                eps_param = pyro.param(
                    "eps" + self.param_suffix,
                    torch.tensor(self.noise_prior),
                    constraint=constraints.positive,
                )
                if self.type_noise:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
                else:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param))

            # Plate on experiments and sample individual values
            with pyro.plate("trials", exp_data.shape[2]):
                for (name, val, dim) in zip(self.names, self.loc_loc_priors, self.dims):
                    # Fix this to init to the mean (or a sample I guess)
                    ll_param = pyro.param(
                        name + self.param_suffix,
                        torch.zeros_like(val)
                        .unsqueeze(0)
                        .repeat((exp_data.shape[2],) + (1,) * dim)
                        + 0.5,
                        constraint=constraints.interval(0.0, 1.0),
                    )
                    param_value = pyro.sample(name, dist.Delta(ll_param).to_event(dim))

        self.extra_param_names = [var + self.param_suffix for var in self.names]

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with pyro.plate("trials", exp_data.shape[2]):
            # Sample the bottom level parameters
            bmodel = self.maker(
                *self.sample_bot(), extra_params=self.get_extra_params()
            )
            # Generate the results
            predictions = bmodel.solve_both(
                exp_data[0], exp_data[1], exp_data[2], exp_control
            )
            # Process the results
            results = experiments.convert_results(
                predictions[:, :, 0], exp_cycles, exp_types
            )

            # Sample!
            with pyro.plate("time", exp_data.shape[1]):
                pyro.sample(
                    "obs",
                    dist.Normal(results * self.new_weights[None, :], full_noise),
                    obs=exp_results * self.new_weights[None, :],
                )

        return results


class HierarchicalNewScaleModel(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        new_weights,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise
        self.new_weights = new_weights
        self.names = names

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.Normal(loc_loc, loc_scale).to_event(dim)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.HalfNormal(scale_scale).to_event(dim)),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroSample(
                    lambda self, var=var, dim=loc_loc.dim(): dist.Normal(
                        getattr(self, var + loc_suffix),
                        getattr(self, var + scale_suffix),
                    ).to_event(dim)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
            else:
                self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        self.extra_param_names = []

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
            # Setup and sample the top-level loc and scale
            top_loc_samples = []
            top_scale_samples = []
            for var, loc_loc, loc_scale, scale_scale, in zip(
                self.names,
                self.loc_loc_priors,
                self.loc_scale_priors,
                self.scale_scale_priors,
            ):
                dim = loc_loc.dim()
                loc_param = pyro.param(
                    var + self.loc_suffix + self.param_suffix,
                    loc_loc,
                    constraint=constraints.interval(0.0, 1.0),
                )
                scale_param = pyro.param(
                    var + self.scale_suffix + self.param_suffix,
                    scale_scale,
                    constraint=constraints.positive,
                )

                top_loc_samples.append(
                    pyro.sample(
                        var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)
                    )
                )
                top_scale_samples.append(
                    pyro.sample(
                        var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)
                    )
                )

            # Add in the noise, if included in the inference
            if self.include_noise:
                eps_param = pyro.param(
                    "eps" + self.param_suffix,
                    torch.tensor(self.noise_prior),
                    constraint=constraints.positive,
                )
                if self.type_noise:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
                else:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param))

            # Plate on experiments and sample individual values
            with poutine.scale(scale=self.new_weights):
                with pyro.plate("trials", exp_data.shape[2]):
                    for (name, val, dim) in zip(
                        self.names, self.loc_loc_priors, self.dims
                    ):
                        # Fix this to init to the mean (or a sample I guess)
                        ll_param = pyro.param(
                            name + self.param_suffix,
                            torch.zeros_like(val)
                            .unsqueeze(0)
                            .repeat((exp_data.shape[2],) + (1,) * dim)
                            + 0.5,
                            constraint=constraints.interval(0.0, 1.0),
                        )
                        param_value = pyro.sample(
                            name, dist.Delta(ll_param).to_event(dim)
                        )

        self.extra_param_names = [var + self.param_suffix for var in self.names]

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with poutine.scale(scale=self.new_weights):
            with pyro.plate("trials", exp_data.shape[2]):
                # Sample the bottom level parameters
                bmodel = self.maker(
                    *self.sample_bot(), extra_params=self.get_extra_params()
                )
                # Generate the results
                predictions = bmodel.solve_both(
                    exp_data[0], exp_data[1], exp_data[2], exp_control
                )
                # Process the results
                results = experiments.convert_results(
                    predictions[:, :, 0], exp_cycles, exp_types
                )

                # Sample!
                with pyro.plate("time", exp_data.shape[1]):
                    pyro.sample(
                        "obs",
                        dist.Normal(results, full_noise),
                        obs=exp_results,
                    )

        return results


class HierarchicalMLE(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise

        self.names = names

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroParam(loc_loc, constraint=constraints.interval(0.0, 1.0)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroParam(scale_scale, constraint=constraints.positive),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroParam(
                    torch.zeros_like(loc_loc), constraint=constraints.interval(0.0, 1.0)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroParam(
                    torch.ones_like(noise_prior), constraint=constraints.positive
                )
            else:
                self.eps = PyroParam(
                    torch.ones_like(noise_prior), constraint=constraints.positive
                )
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        # self.extra_param_names = []
        self.extra_param_names = [var for var in self.names]

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
            # Setup and sample the top-level loc and scale
            pass

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with pyro.plate("trials", exp_data.shape[2]):
            # Sample the bottom level parameters
            bmodel = self.maker(
                *self.sample_bot(), extra_params=self.get_extra_params()
            )
            # Generate the results
            predictions = bmodel.solve_both(
                exp_data[0], exp_data[1], exp_data[2], exp_control
            )
            # Process the results
            results = experiments.convert_results(
                predictions[:, :, 0], exp_cycles, exp_types
            )

            # Sample!
            with pyro.plate("time", exp_data.shape[1]):
                pyro.sample("obs", dist.Normal(results, full_noise), obs=exp_results)

        return results


class HierarchicalRegularizerModel(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        reg_suffix="regular",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise
        self.reg_suffix = reg_suffix

        self.names = names

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.Normal(loc_loc, loc_scale).to_event(dim)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.HalfNormal(scale_scale).to_event(dim)),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroSample(
                    lambda self, var=var, dim=loc_loc.dim(): dist.Normal(
                        getattr(self, var + loc_suffix),
                        getattr(self, var + scale_suffix),
                    ).to_event(dim)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
            else:
                self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        self.extra_param_names = []

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def L2_regularizer(self, my_parameters, lam=torch.tensor(1.0)):
        """
        Define the L2 regularizer
        """
        reg_loss = 0.0
        for param in my_parameters:
            reg_loss = reg_loss + param.pow(2.0).sum()
        return lam * reg_loss

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
            # Setup and sample the top-level loc and scale
            top_loc_samples = []
            top_scale_samples = []
            for var, loc_loc, loc_scale, scale_scale, in zip(
                self.names,
                self.loc_loc_priors,
                self.loc_scale_priors,
                self.scale_scale_priors,
            ):
                dim = loc_loc.dim()
                loc_param = pyro.param(
                    var + self.loc_suffix + self.param_suffix,
                    loc_loc,
                    constraint=constraints.interval(0.0, 1.0),
                )
                scale_param = pyro.param(
                    var + self.scale_suffix + self.param_suffix,
                    scale_scale,
                    constraint=constraints.positive,
                )

                top_loc_samples.append(
                    pyro.sample(
                        var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)
                    )
                )
                top_scale_samples.append(
                    pyro.sample(
                        var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)
                    )
                )

            # Add in the noise, if included in the inference
            if self.include_noise:
                eps_param = pyro.param(
                    "eps" + self.param_suffix,
                    torch.tensor(self.noise_prior),
                    constraint=constraints.positive,
                )
                if self.type_noise:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
                else:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param))

            # Plate on experiments and sample individual values
            with pyro.plate("trials", exp_data.shape[2]):
                for (name, val, dim) in zip(self.names, self.loc_loc_priors, self.dims):
                    # Fix this to init to the mean (or a sample I guess)
                    ll_param = pyro.param(
                        name + self.param_suffix,
                        torch.zeros_like(val)
                        .unsqueeze(0)
                        .repeat((exp_data.shape[2],) + (1,) * dim)
                        + 0.5,
                        constraint=constraints.interval(0.0, 1.0),
                    )
                    pyro.factor(
                        name + self.reg_suffix,
                        self.L2_regularizer(ll_param),
                        has_rsample=True,
                    )
                    param_value = pyro.sample(name, dist.Delta(ll_param).to_event(dim))

        self.extra_param_names = [var + self.param_suffix for var in self.names]

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results=None):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with pyro.plate("trials", exp_data.shape[2]):
            # Sample the bottom level parameters
            bmodel = self.maker(
                *self.sample_bot(), extra_params=self.get_extra_params()
            )
            # Generate the results
            predictions = bmodel.solve_both(
                exp_data[0], exp_data[1], exp_data[2], exp_control
            )
            # Process the results
            results = experiments.convert_results(
                predictions[:, :, 0], exp_cycles, exp_types
            )

            # Sample!
            with pyro.plate("time", exp_data.shape[1]):
                pyro.sample("obs", dist.Normal(results, full_noise), obs=exp_results)

        return results


# define VAE based inference model for extrapolate data


class ExtrapNN(nn.Module):
    def __init__(self, sdata, fdata, hidden_dim):
        super().__init__()
        self.batch_size = sdata.shape[1]
        # setup the two linear transformations used
        self.fc1 = nn.Linear(sdata.shape[0] * sdata.shape[1], hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, hidden_dim)
        self.fc32 = nn.Linear(hidden_dim, fdata.shape[0] * fdata.shape[1])
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        hidden1 = F.relu(self.fc1(data.T.flatten(0)))
        hidden2 = F.relu(self.fc21(hidden1))
        x = self.fc32(hidden2)
        return x.reshape(self.batch_size, int(x.shape[0] / self.batch_size)).T


class HierarchicalExtrapModel(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        surrogate_model,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise

        self.names = names

        # Store the surrogate model
        self.surrogate_model = surrogate_model

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.Normal(loc_loc, loc_scale).to_event(dim)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.HalfNormal(scale_scale).to_event(dim)),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroSample(
                    lambda self, var=var, dim=loc_loc.dim(): dist.Normal(
                        getattr(self, var + loc_suffix),
                        getattr(self, var + scale_suffix),
                    ).to_event(dim)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
            else:
                self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        self.extra_param_names = []

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(
            exp_data, exp_cycles, exp_types, exp_control, exp_results, full_results=None
        ):
            # Setup and sample the top-level loc and scale
            top_loc_samples = []
            top_scale_samples = []
            for var, loc_loc, loc_scale, scale_scale, in zip(
                self.names,
                self.loc_loc_priors,
                self.loc_scale_priors,
                self.scale_scale_priors,
            ):
                dim = loc_loc.dim()
                loc_param = pyro.param(
                    var + self.loc_suffix + self.param_suffix,
                    loc_loc,
                    constraint=constraints.interval(0.0, 1.0),
                )
                scale_param = pyro.param(
                    var + self.scale_suffix + self.param_suffix,
                    scale_scale,
                    constraint=constraints.positive,
                )

                top_loc_samples.append(
                    pyro.sample(
                        var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)
                    )
                )
                top_scale_samples.append(
                    pyro.sample(
                        var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)
                    )
                )

            # Add in the noise, if included in the inference
            if self.include_noise:
                eps_param = pyro.param(
                    "eps" + self.param_suffix,
                    torch.tensor(self.noise_prior),
                    constraint=constraints.positive,
                )
                if self.type_noise:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
                else:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param))

            # Plate on experiments and sample individual values
            with pyro.plate("trials", exp_data.shape[2]):
                for (name, val, dim) in zip(self.names, self.loc_loc_priors, self.dims):
                    # Fix this to init to the mean (or a sample I guess)
                    ll_param = pyro.param(
                        name + self.param_suffix,
                        torch.zeros_like(val)
                        .unsqueeze(0)
                        .repeat((exp_data.shape[2],) + (1,) * dim)
                        + 0.5,
                        constraint=constraints.interval(0.0, 1.0),
                    )
                    param_value = pyro.sample(name, dist.Delta(ll_param).to_event(dim))

        self.extra_param_names = [var + self.param_suffix for var in self.names]

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(
        self,
        exp_data,
        exp_cycles,
        exp_types,
        exp_control,
        exp_results,
        full_results=None,
    ):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with pyro.plate("trials", exp_data.shape[2]):
            # Sample the bottom level parameters
            bmodel = self.maker(
                *self.sample_bot(), extra_params=self.get_extra_params()
            )
            # Generate the results
            predictions = bmodel.solve_both(
                exp_data[0], exp_data[1], exp_data[2], exp_control
            )
            # Process the results
            results = experiments.convert_results(
                predictions[:, :, 0], exp_cycles, exp_types
            )
            mean_results = self.surrogate_model(results)

            # Sample!
            with pyro.plate("time", full_results.shape[0]):
                pyro.sample(
                    "obs", dist.Normal(mean_results, full_noise), obs=full_results
                )

        return results


class Interp1d(torch.autograd.Function):
    # use the interpolate function developed here:
    # https://github.com/aliutkus/torchinterp1d/blob/master/torchinterp1d/interp1d.py
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
            assert len(vec.shape) <= 2, "interp1d: all inputs must be " "at most 2-D."
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, "All parameters must be on the same device."
        device = device[0]

        # Checking for the dimensions
        assert v["x"].shape[1] == v["y"].shape[1] and (
            v["x"].shape[0] == v["y"].shape[0]
            or v["x"].shape[0] == 1
            or v["y"].shape[0] == 1
        ), (
            "x and y must have the same number of columns, and either "
            "the same number of row or one of them having only one "
            "row."
        )

        reshaped_xnew = False
        if (
            (v["x"].shape[0] == 1)
            and (v["y"].shape[0] == 1)
            and (v["xnew"].shape[0] > 1)
        ):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v["xnew"].shape
            v["xnew"] = v["xnew"].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v["x"].shape[0], v["xnew"].shape[0])

        shape_ynew = (D, v["xnew"].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0] * shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v["xnew"].shape[0] == 1:
            v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

        torch.searchsorted(v["x"].contiguous(), v["xnew"].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ["x", "y", "xnew"]:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [
                    None,
                ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat["slopes"] = is_flat["x"]

        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
                eps + (v["x"][:, 1:] - v["x"][:, :-1])
            )

            # now build the linear interpolation
            ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
            ctx.saved_tensors[0],
            [i for i in inputs if i is not None],
            grad_out,
            retain_graph=True,
        )
        result = [
            None,
        ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class HierarchicalDenseModel(PyroModule):
    """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args:
      maker (function):                     function that returns a valid
                                            :py:class`torch.nn.Module`, given the input
                                            parameters
      names (list(str)):                    names to use for the parameters
      loc_loc_priors (list(tensor)):        location of the prior for the mean
                                            of each parameter
      loc_scale_priors (list(tensor)):      scale of the prior of the mean of each
                                            parameter
      scale_scale_priors (list(tensor)):    scale of the prior of the standard
                                            deviation of each parameter
      noise_priors (list or scalar):        random noise, can be either a single scalar
                                            or a 1D tensor if it's a 1D tensor then each
                                            entry i represents the noise in test type i

    Keyword Args:
      loc_suffix (str):                     append to the variable name to give the top
                                            level sample for the location, default :code:`"_loc"`
      scale_suffix (str):                   append to the variable name to give the top
                                            level sample for the scale, default :code:`"_scale"`
      param_suffix (str):                   append to the variable name to give the corresponding
                                            :py:mod:`pyro.param` name, default
                                            :code:`"_param"`
      include_noise (str):                  if :code:`True` include white noise in the inference,
                                            default :code:`False`
    """

    def __init__(
        self,
        maker,
        names,
        loc_loc_priors,
        loc_scale_priors,
        scale_scale_priors,
        noise_prior,
        loc_suffix="_loc",
        scale_suffix="_scale",
        param_suffix="_param",
        include_noise=False,
    ):
        super().__init__()

        # Store things we might later
        self.maker = maker
        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix
        self.param_suffix = param_suffix
        self.include_noise = include_noise

        self.names = names

        # We need these for the shapes...
        self.loc_loc_priors = loc_loc_priors
        self.loc_scale_priors = loc_scale_priors
        self.scale_scale_priors = scale_scale_priors
        self.noise_prior = noise_prior

        # What type of noise are we using
        self.type_noise = noise_prior.dim() > 0

        # Setup both the top and bottom level variables
        self.bot_vars = names
        self.top_vars = []
        self.dims = []
        for (
            var,
            loc_loc,
            loc_scale,
            scale_scale,
        ) in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
            # These set standard PyroSamples with names of var + suffix
            dim = loc_loc.dim()
            self.dims.append(dim)
            self.top_vars.append(var + loc_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.Normal(loc_loc, loc_scale).to_event(dim)),
            )
            self.top_vars.append(var + scale_suffix)
            setattr(
                self,
                self.top_vars[-1],
                PyroSample(dist.HalfNormal(scale_scale).to_event(dim)),
            )

            # The tricks are: 1) use lambda self and 2) remember how python binds...
            setattr(
                self,
                var,
                PyroSample(
                    lambda self, var=var, dim=loc_loc.dim(): dist.Normal(
                        getattr(self, var + loc_suffix),
                        getattr(self, var + scale_suffix),
                    ).to_event(dim)
                ),
            )

        # Setup the noise
        if self.include_noise:
            if self.type_noise:
                self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
            else:
                self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.eps = noise_prior

        # This annoyance is required to make the adjoint solver work
        self.extra_param_names = []

    @property
    def nparams(self):
        """
        Number of parameters in model
        """
        return len(self.names)

    def sample_top(self):
        """
        Sample the top level variables
        """
        return [getattr(self, name) for name in self.top_vars]

    def sample_bot(self):
        """
        Sample the bottom level variables
        """
        return [getattr(self, name) for name in self.bot_vars]

    def make_guide(self):
        # pylint: disable=unused-variable
        """
        Make the guide and cache the extra parameter names the adjoint solver
        is going to need
        """

        def guide(
            exp_data,
            exp_cycles,
            exp_types,
            exp_control,
            exp_results,
            full_data,
            full_cycles,
            full_types,
            full_control,
            full_results=None,
        ):
            # Setup and sample the top-level loc and scale
            top_loc_samples = []
            top_scale_samples = []
            for var, loc_loc, loc_scale, scale_scale, in zip(
                self.names,
                self.loc_loc_priors,
                self.loc_scale_priors,
                self.scale_scale_priors,
            ):
                dim = loc_loc.dim()
                loc_param = pyro.param(
                    var + self.loc_suffix + self.param_suffix,
                    loc_loc,
                    constraint=constraints.interval(0.0, 1.0),
                )
                scale_param = pyro.param(
                    var + self.scale_suffix + self.param_suffix,
                    scale_scale,
                    constraint=constraints.positive,
                )

                top_loc_samples.append(
                    pyro.sample(
                        var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)
                    )
                )
                top_scale_samples.append(
                    pyro.sample(
                        var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)
                    )
                )

            # Add in the noise, if included in the inference
            if self.include_noise:
                eps_param = pyro.param(
                    "eps" + self.param_suffix,
                    torch.tensor(self.noise_prior),
                    constraint=constraints.positive,
                )
                if self.type_noise:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
                else:
                    eps_sample = pyro.sample("eps", dist.Delta(eps_param))

            # Plate on experiments and sample individual values
            with pyro.plate("trials", exp_data.shape[2]):
                for (name, val, dim) in zip(self.names, self.loc_loc_priors, self.dims):
                    # Fix this to init to the mean (or a sample I guess)
                    ll_param = pyro.param(
                        name + self.param_suffix,
                        torch.zeros_like(val)
                        .unsqueeze(0)
                        .repeat((exp_data.shape[2],) + (1,) * dim)
                        + 0.5,
                        constraint=constraints.interval(0.0, 1.0),
                    )
                    param_value = pyro.sample(name, dist.Delta(ll_param).to_event(dim))

        self.extra_param_names = [var + self.param_suffix for var in self.names]

        return guide

    def get_extra_params(self):
        """
        Actually list the extra parameters required for the adjoint solve.

        We can't determine this by introspection on the base model, so
        it needs to be done here
        """
        # Do some consistency checking
        for p in self.extra_param_names:
            if p not in pyro.get_param_store().keys():
                raise ValueError(f"Internal error, parameter {p} not in store!")

        return [pyro.param(name).unconstrained() for name in self.extra_param_names]

    def forward(
        self,
        exp_data,
        exp_cycles,
        exp_types,
        exp_control,
        exp_results,
        full_data,
        full_cycles,
        full_types,
        full_control,
        full_results=None,
    ):
        # pylint: disable=unused-variable
        """
        Evaluate the forward model, optionally conditioned by the experimental
        data.

        Optionally condition on the actual data

        See the :py:mod:`pyoptmat.experiments` module
        for detailed on how to format the input to this function

        Args:
          exp_data (torch.tensor):    formatted input experimental data
          exp_cycles (torch.tensor):  cycle counts for each test
          exp_types (torch.tensor):   experiment types, as integers
          exp_control (torch.tensor): stress/strain control flag

        Keyword Args:
          exp_results (torch.tensor): true results for conditioning
        """
        # Sample the top level parameters
        curr = self.sample_top()
        eps = self.eps

        # Setup the full noise, which can be type specific
        if self.type_noise:
            full_noise = torch.empty(exp_data.shape[-1], device=exp_data.device)
            for i in experiments.exp_map.values():
                full_noise[exp_types == i] = eps[i]
        else:
            full_noise = eps

        with pyro.plate("trials", exp_data.shape[2]):
            # Sample the bottom level parameters
            bmodel = self.maker(
                *self.sample_bot(), extra_params=self.get_extra_params()
            )
            # Generate the results
            predictions = bmodel.solve_both(
                exp_data[0], exp_data[1], exp_data[2], exp_control
            )

            # interpolate to dense data
            old_x = torch.linspace(0, 1, exp_data.shape[1]).repeat(
                (exp_data.shape[2],) + (1,)
            )
            new_x = torch.linspace(0, 1, full_data.shape[1])

            interp_model = Interp1d(old_x, predictions[:, :, 0].T, new_x)
            mean_results = interp_model(old_x, predictions[:, :, 0].T, new_x).T
            # Process the results
            results = experiments.convert_results(mean_results, full_cycles, full_types)

            # Sample!
            with pyro.plate("time", full_results.shape[0]):
                pyro.sample("obs", dist.Normal(results, full_noise), obs=full_results)

        return results
