from __future__ import annotations

from functools import wraps

import torch
from torch import nn
from torch import distributions as dist
from torch.nn.functional import softplus
from torch.distributions import constraints
from torch.distributions.utils import probs_to_logits, logits_to_probs, broadcast_all

from .utils import to_one_hot, flatten


# TODO canonical params -> invert transform params


def get_likelihood(*dist_names, flatten=True):
    assert len(dist_names) > 0

    cls = LikelihoodFlatten if flatten else LikelihoodList

    if len(dist_names) == 1:
        if isinstance(dist_names[0], str):
            return get_distribution_by_name(dist_names[0])
        elif isinstance(dist_names[0], dict):
            return get_distribution_by_name(**dist_names[0])
        elif len(dist_names[0]) == 1:
            return cls(get_likelihood(*dist_names[0], flatten=False))
        else:
            return get_likelihood(*dist_names[0], flatten=False)

    dist_list = []
    for name in dist_names:
        if not isinstance(name, str):  # Iterable with strings
            dist_list.append(get_likelihood(name, flatten=False))
        else:
            dist_list.append(get_distribution_by_name(name))

    return cls(*dist_list)


def get_distribution_by_name(name, *, domain_size=1, ensure_args=True):
    is_gammatrick = name[-1] == '*'
    is_berntrick = name[-1] == '+' and 'categorical' in name
    size = 0
    available_dists = {
        'normal': Normal, 'lognormal': LogNormal, 'gamma': Gamma, 'exponential': Exponential,
        'bernoulli': Bernoulli, 'poisson': Poisson, 'categorical': Categorical, 'one-hot-categorical': OneHotCategorical
    }

    if is_gammatrick or is_berntrick:
        name = name[:-1]

    if 'categorical' in name:
        pos = name.find('(')
        size = int(name[pos + 1: name.find(')')])
        name = name[:pos]

    if is_gammatrick and 'categorical' in name:
        requested_dist = GammaTrickCategorical(size, ensure_args=ensure_args)
    elif is_berntrick:
        requested_dist = BernoulliTrickCategorical(size, ensure_args=ensure_args)
    else:
        if size > 0:
            requested_dist = available_dists[name](size, ensure_args=ensure_args)
        else:
            requested_dist = available_dists[name](domain_size=domain_size, ensure_args=ensure_args)
        if is_gammatrick:
            requested_dist = GammaTrick(requested_dist, ensure_args=ensure_args)

    return requested_dist


def ensure(value, constraint, noise=1e-15):
    if constraint == constraints.positive:  # Special case for the rate of the poisson distribution
        value = torch.clamp(value, min=noise)

    elif isinstance(constraint, constraints.greater_than):
        lower_bound = constraint.lower_bound
        value = lower_bound + noise + softplus(value)

    elif isinstance(constraint, constraints.less_than):
        upper_bound = constraint.upper_bound
        value = upper_bound - noise - softplus(value)

    elif constraint == constraints.simplex:
        value = logits_to_probs(value)

    elif constraint == constraints.unit_interval:
        value = torch.clamp(value, min=0., max=1.)

    return value


class BaseLikelihood(nn.Module):
    r"""
    BaseLikelihood is the abstract base class for any likelihood function.

    In contrast with distributions from torch.distribution, these ones act as Modules with memory to keep track of the
    scale of the preprocessing step, and whether the model is in training on evaluation mode.
    """

    arg_constraints = {}
    is_discrete = False

    def _ensure_params(self, func):
        def _ensure_params_(*params):
            assert len(params) == self.num_params, f'{len(params)} v.s. {self.num_params} {self}'
            params = list(flatten(params))
            new_params = broadcast_all(*params)
            self._counter += 1

            if self._counter == 1 and self.ensure_args:
                new_params = [ensure(x, c) for x, c in zip(new_params, self.arg_constraints)]

            result = func(*new_params)
            self._counter -= 1
            return result
        return _ensure_params_

    def __init__(self, *, domain_size=1, ensure_args=True):
        super(BaseLikelihood, self).__init__()
        self._domain_size = domain_size
        # self._scale = torch.ones([domain_size])
        self.register_buffer('_scale', torch.ones([domain_size]))
        self.ensure_args = ensure_args

        self._counter = 0
        self.transform_params = self._ensure_params(self.transform_params)
        self.inverse_transform_params = self._ensure_params(self.inverse_transform_params)
        self.canonical_params = self._ensure_params(self.canonical_params)
        self.instantiate = self._ensure_params(self.instantiate)
        self.get_params = self._ensure_params(self.get_params)

    @property
    def dist(self) -> type(dist.Distribution):
        raise NotImplementedError

    @property
    def num_params(self):
        return len(self.arg_constraints)

    @property
    def domain_size(self):
        return self._domain_size

    @property
    def scale(self):
        """
        Returns the scale used to scaled the data or scaled back the parameters.
        """
        return self._scale if not self.is_discrete else torch.ones_like(self._scale)

    @scale.setter
    def scale(self, value):
        if (not torch.is_tensor(value) or value.numel() == 1) and self.domain_size == 1:
            value = torch.ones_like(self.scale) * value
        elif not torch.is_tensor(value):
            value = torch.tensor(value)  # e.g. numpy tensor

        assert value.size() == torch.Size([self.domain_size]), value.size()
        self._scale = value

    def transform_data(self, x):
        return x * self.scale

    def inverse_transform_data(self, x):
        return x / self.scale

    def transform_params(self, *params):
        raise NotImplementedError

    def inverse_transform_params(self, *params):
        raise NotImplementedError

    def get_params(self, *params):
        return params

    def instantiate(self, *params):
        def scale_input(func):
            @wraps(func)
            def scale_input_(*args, **kwargs):  # TODO Fix aliasing
                if len(args) > 0:
                    args = list(args)
                    value = args.pop(0)
                    value = self.transform_data(value)
                    return func(value, *args[1:], **kwargs)
                elif len(kwargs) == 1:
                    key = next(iter(kwargs))
                    value = kwargs.pop(key)
                    value = self.transform_data(value)
                    return func(**{key: value}, **kwargs)
                else:
                    raise Exception

            return scale_input_

        if not self.training:
            params = self.inverse_transform_params(*params)

        instance = self.dist(**self.canonical_params(*params))

        if self.training:
            # Decorate the functions so it transparently scales the data
            instance.log_prob = scale_input(instance.log_prob)
            instance.cdf = scale_input(instance.cdf)
            instance.icdf = scale_input(instance.icdf)

        return instance

    def canonical_params(self, *params):
        raise NotImplementedError

    @property
    def canonical_constraints(self):
        return self.dist.arg_constraints

    def compute_hessian(self, data):
        raise NotImplementedError

    def compute_lipschitz(self, data, original_hessian):
        raise NotImplementedError

    def __call__(self, *params):
        return self.instantiate(*params)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        assert item == 0
        return self

    def __rshift__(self, data):
        return self.transform_data(data)

    def __lshift__(self, data):
        return self.inverse_transform_data(data)

    @property
    def InputScaler(self):
        class DataScaler(nn.Module):
            def __init__(self, obj):
                super(DataScaler, self).__init__()

                self.obj = obj

            def forward(self, data):
                return self.obj >> data

        return DataScaler(self)


def _apply(counter_function, as_tensor):
    def _apply_(func):
        def _apply__(self, *value):
            if as_tensor:
                value = value[0]

            new_value, pos = [], 0
            for d in self.dist_list:
                if as_tensor:
                    new_value.append(getattr(d, func.__name__)(value[..., pos: pos + counter_function(d)]))
                else:
                    new_value.append(getattr(d, func.__name__)(*value[pos: pos + counter_function(d)]))
                pos += counter_function(d)
            if as_tensor:
                assert pos == value.size(-1), f'stopped at {pos} instead of {value.size(-1)}'
            else:
                assert pos == len(value), f'stopped at {pos} instead of {len(value)}'
            return torch.cat(new_value, dim=-1) if as_tensor else new_value
        return _apply__
    return _apply_


class LikelihoodList(BaseLikelihood):
    def _ensure_params(self, func):
        def _ensure_params_(*params):
            return func(*params)
        return _ensure_params_

    def __init__(self, *dist_list, ensure_args=True):
        assert len(dist_list) > 0, f'Nothing passed to {type(self).__name__}!'

        if len(dist_list) == 1 and not isinstance(dist_list[0], BaseLikelihood):
            dist_list = dist_list[0]
        assert all([isinstance(d, BaseLikelihood) for d in dist_list]), f'Passed a non-distribution to ' \
                                                                        f'{type(self).__name__}'

        self.arg_constraints = sum([d.arg_constraints for d in dist_list], [])

        super(LikelihoodList, self).__init__(ensure_args=ensure_args)
        self.dist_list = torch.nn.ModuleList(dist_list)

        del self._scale  # Ensure we do not use it

    @property
    def dist(self) -> type(dist.Distribution):
        raise NotImplementedError

    @property
    def is_discrete(self):
        return all([d.is_discrete for d in self])  # This may change in runtime

    @property
    def domain_size(self):
        return sum([d.domain_size for d in self])  # This may change in runtime

    @property
    def canonical_constraints(self):
        return flatten([d.canonical_constraints for d in self.dist_list])

    @property
    def scale(self):  # It is intended to not set up the setter method
        return [d.scale for d in self.dist_list]

    @_apply(lambda d: d.domain_size, as_tensor=True)
    def transform_data(self, x):
        pass

    @_apply(lambda d: d.domain_size, as_tensor=True)
    def inverse_transform_data(self, x):
        pass

    @_apply(lambda d: d.num_params, as_tensor=False)
    def transform_params(self, *params):
        pass

    @_apply(lambda d: d.num_params, as_tensor=False)
    def inverse_transform_params(self, *params):
        pass

    @_apply(lambda d: d.num_params, as_tensor=False)
    def canonical_params(self, *params):
        pass

    def instantiate(self, *params):
        from .utils import Wrapper

        class ListWrapper(Wrapper):
            __wraps__ = list
            __parent__ = self

            def __getattr__(self, item):
                def apply(func_name):
                    def apply_(*args, **kwargs):
                        value = None
                        if 'value' in kwargs:
                            value = kwargs.pop('value')
                        elif len(args) > 0 and torch.is_tensor(args[0]):
                            args = list(args)
                            value = args.pop(0)

                        result, pos = [], 0
                        for d, l in zip(self, self.__parent__):
                            if value is not None:
                                size = l.domain_size

                                value_d = value[..., pos: pos+size]
                                result_d = getattr(d, func_name)(*args, value=value_d.squeeze(dim=-1), **kwargs)
                                if item == 'log_prob' and isinstance(d, dist.Distribution):  # special case
                                    result_d = result_d.unsqueeze(dim=-1)
                                result.append(result_d)
                                pos += size
                            else:
                                result_d = getattr(d, func_name)
                                if callable(result_d):
                                    result_d = result_d(*args, **kwargs)

                                # special cases
                                if isinstance(d, dist.Distribution):
                                    if d.event_shape == torch.Size([]) and item in ['sample', 'mean']:
                                        result_d = result_d.unsqueeze(dim=-1)
                                    if item in ['entropy']:  # special cases
                                        result_d = result_d.unsqueeze(dim=-1)

                                result.append(result_d)

                        if len(result) > 0 and torch.is_tensor(result[0]):
                            result = torch.cat([x.float() for x in result], dim=-1)

                        return result
                    return apply_

                try:
                    return super(ListWrapper, self).__getattr__(item)
                except AttributeError as e:
                    if len(self) > 0:
                        if callable(getattr(self[0], item, None)):
                            return apply(item)
                        else:
                            return property(apply(item)).__get__(self)

                    raise e

        new_value, pos = [], 0
        for d in self.dist_list:
            new_value.append(d.instantiate(*params[pos: pos + d.num_params]))
            pos += d.num_params
        return ListWrapper(new_value)

    def compute_hessian(self, data):
        raise NotImplementedError

    def compute_lipschitz(self, data, original_hessian):
        raise NotImplementedError

    def __len__(self):
        return len(self.dist_list)

    def __getitem__(self, item):
        return self.dist_list[item]

    def __str__(self):
        return ', '.join([str(d) for d in self])


class LikelihoodFlatten(LikelihoodList):
    def __init__(self, *dist_list, ensure_args=True):
        super(LikelihoodFlatten, self).__init__(*dist_list, ensure_args=ensure_args)
        self.flatten = True

        def build_index(dist_list, prefix=tuple()):
            index = []
            for i, d in enumerate(dist_list):
                if not isinstance(d, LikelihoodList):
                    index.append(prefix + (i,))
                else:
                    index += build_index(d, prefix + (i,))
            return index
        self.indexes = build_index(list(dist_list))

    def __len__(self):
        return len(self.indexes) if self.training and self.flatten else super().__len__()

    def __getitem__(self, item):
        if not self.training or not self.flatten:
            return super().__getitem__(item)

        result = self
        for index in self.indexes[item]:
            result = result.dist_list[index]
        return result


class ExponentialFamily(BaseLikelihood):
    @property
    def dist(self) -> type(dist.Distribution):
        raise NotImplementedError

    def canonical_params(self, *params):
        raise NotImplementedError

    @property
    def scale_factors(self):  # TODO
        raise NotImplementedError

    def params_from_data(self, x):
        raise NotImplementedError

    def transform_params(self, *params):
        new_params = []
        for p, f in zip(params, self.scale_factors):
            new_params.append(p / f(self.scale))
        return new_params

    def inverse_transform_params(self, *params):
        new_params = []
        for p, f in zip(params, self.scale_factors):
            new_params.append(p * f(self.scale))
        return new_params

    def compute_hessian(self, data):
        from torch.autograd import grad
        old_value = self.ensure_args
        self.ensure_args = False

        params = self.params_from_data(self >> data)
        params = torch.stack(params)
        params.requires_grad = True

        log_prob = self(*params).log_prob(data).mean()  # for the exponential family
        jacobian = grad(log_prob, params, retain_graph=True, create_graph=True)[0]

        hessian_rows = []
        for jac_i in jacobian:
            row_i = grad(jac_i, params, retain_graph=True, create_graph=False)[0]
            hessian_rows.append(row_i.detach())

        self.ensure_args = old_value
        return hessian_rows

    def compute_lipschitz(self, data, original_hessian=None):
        if original_hessian is None:
            original_hessian = self.compute_hessian(data)
            lipschitzs = [x.norm(p=2) for x in original_hessian]
        else:
            lipschitzs = []
            for i, f_i in enumerate(self.scale_factors):
                row_i = torch.empty(self.num_params)

                for j, f_j in enumerate(self.scale_factors):
                    row_i[j] = original_hessian[i][j] * f_j(self.scale)

                lipschitzs.append(abs(f_i(self.scale)) * row_i.norm(p=2))

        return lipschitzs, original_hessian

##############


class Normal(ExponentialFamily):
    arg_constraints = [
        constraints.real,  # eta1
        constraints.less_than(0)  # eta2
    ]

    @property
    def dist(self) -> type(dist.Distribution):
        return dist.Normal

    def canonical_params(self, *params):
        eta1, eta2 = params
        return {'loc': -0.5 * eta1 / eta2, 'scale': torch.sqrt(-0.5 / eta2)}

    @property
    def scale_factors(self):
        return [lambda w: w, lambda w: w**2]

    def params_from_data(self, x):
        loc, std = x.mean(), x.std()

        eta2 = -0.5 / std ** 2
        eta1 = -2 * loc * eta2

        return eta1, eta2

    def compute_hessian(self, data):
        old_value = self.ensure_args
        self.ensure_args = False

        params = self.params_from_data(self >> data)
        params = self.canonical_params(*params)

        row_1 = torch.tensor([-params['scale']**2, -2 * params['loc'] * params['scale']**2])
        row_2 = torch.tensor([row_1[1], -2 * params['scale']**2 * (params['scale']**2 + params['loc']**2)])

        self.ensure_args = old_value
        return row_1, row_2

    def __str__(self):
        return 'normal'


class LogNormal(Normal):
    @property
    def dist(self) -> type(dist.Distribution):
        return dist.LogNormal

    def transform_data(self, x):
        return torch.clamp(torch.pow(x, self.scale), min=1e-15)  # To ensure that it is positive

    def inverse_transform_data(self, x):
        return torch.pow(x, 1./self.scale)

    def params_from_data(self, x):
        return super(LogNormal, self).params_from_data(torch.log(x))

    def __str__(self):
        return 'log-normal'


class Gamma(ExponentialFamily):
    arg_constraints = [
        constraints.greater_than(-1),  # eta1
        constraints.less_than(0)  # eta2
    ]

    @property
    def dist(self) -> type(dist.Distribution):
        return dist.Gamma

    def canonical_params(self, *params):
        eta1, eta2 = params

        return {'concentration': eta1 + 1, 'rate': -eta2}

    @property
    def scale_factors(self):
        return [lambda w: torch.ones_like(w), lambda w: w]

    def params_from_data(self, x):
        mean, meanlog = x.mean(), x.log().mean()
        s = mean.log() - meanlog

        shape = (3 - s + ((s-3)**2 + 24*s).sqrt()) / (12 * s)
        for _ in range(50):
            shape = shape - (shape.log() - torch.digamma(shape) - s) / (1 / shape - torch.polygamma(1, shape))

        concentration = shape
        rate = shape / mean

        eta1 = concentration - 1
        eta2 = -rate

        return eta1, eta2

    def __str__(self):
        return 'gamma'

    def compute_hessian(self, data):
        old_value = self.ensure_args
        self.ensure_args = False

        params = self.params_from_data(self >> data)
        params = self.canonical_params(*params)

        row_1 = torch.tensor([
            (params['concentration'] - 1) * torch.polygamma(1, params['concentration']) - 1.,
            -params['rate'].reciprocal()
        ])
        row_2 = torch.tensor([
            row_1[1],
            -params['concentration'] * params['rate'].pow(2).reciprocal()
        ])

        self.ensure_args = old_value
        return row_1, row_2


class Exponential(ExponentialFamily):
    arg_constraints = [
        constraints.less_than(0)  # eta1
    ]

    @property
    def dist(self):
        return dist.Exponential

    def canonical_params(self, *params):
        return {'rate': -params[0]}

    @property
    def scale_factors(self):
        return [lambda w: w]

    def params_from_data(self, x):
        mean = x.mean()
        return -1 / mean,

    def compute_hessian(self, data):
        old_value = self.ensure_args
        self.ensure_args = False

        params = self.params_from_data(self >> data)
        params = self.canonical_params(*params)

        row = torch.tensor([-params['rate'].pow(2).reciprocal()])

        self.ensure_args = old_value
        return row,

    def __str__(self):
        return 'exponential'


class Bernoulli(ExponentialFamily):
    is_discrete = True
    arg_constraints = [constraints.real]

    @property
    def dist(self) -> type(dist.Distribution):
        return dist.Bernoulli

    def canonical_params(self, *params):
        if self.training:
            return {'logits': params[0]}

        return {'probs': logits_to_probs(params[0], is_binary=True)}

    @property
    def canonical_constraints(self):
        return {'probs': constraints.unit_interval}

    @property
    def scale_factors(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return probs_to_logits(x.mean(), is_binary=True),

    def compute_hessian(self, data):
        raise NotImplementedError

    def __str__(self):
        return 'bernoulli'


class Poisson(ExponentialFamily):
    is_discrete = True
    arg_constraints = [constraints.real]

    @property
    def dist(self) -> type(dist.Distribution):
        return dist.Poisson

    def canonical_params(self, *params):
        return {'rate': torch.exp(params[0])}

    @property
    def scale_factors(self):
        return [lambda w: torch.ones_like(w)]

    def params_from_data(self, x):
        return torch.log(x.mean()),

    def compute_hessian(self, data):
        raise NotImplementedError

    def __str__(self):
        return 'poisson'


class GammaTrick(Gamma):
    def __init__(self, og_dist, noise=None, *, ensure_args=True):
        super(GammaTrick, self).__init__(domain_size=og_dist.domain_size, ensure_args=ensure_args)
        assert isinstance(og_dist, ExponentialFamily) and og_dist.is_discrete, 'Gamma trick is only available for' \
                                                                               ' discrete distributions'
        assert og_dist.num_params == 1, 'The discrete distribution is supposed to be parametrized by its mean'

        self.og_dist = og_dist
        self.noise = noise if noise else dist.Beta(1.1, 30)
        self.constant = 1.

    @property
    def dist(self) -> type(dist.Distribution):
        return super().dist if self.training else self.og_dist.dist

    def canonical_params(self, *params):
        gamma_params = super().canonical_params(*params)
        if self.training:
            return gamma_params

        mean = super().dist(**gamma_params).mean - self.constant - self.noise.mean
        discrete_key, constraint = next(iter(self.og_dist.canonical_constraints.items()))
        discrete_value = ensure(mean, constraint)

        return {discrete_key: discrete_value}

    def transform_data(self, x):
        noise = self.noise.sample(x.size()).to(x.device)
        return super().transform_data(x + self.constant + noise)

    def inverse_transform_data(self, x):
        x = super().inverse_transform_data(x)
        return torch.floor(x - self.constant)

    def __str__(self):
        return super().__str__() if self.training else f'{str(self.og_dist)}*'


class Categorical(ExponentialFamily):
    is_discrete = True

    def __init__(self, size, *, ensure_args=True):
        super(Categorical, self).__init__(domain_size=1, ensure_args=ensure_args)
        self.size = size
        self.arg_constraints = [constraints.real] * size

    @property
    def dist(self) -> type(dist.Distribution):
        return dist.Categorical

    @property
    def canonical_constraints(self):
        return {'probs': constraints.simplex}

    def canonical_params(self, *params):
        key, constraint = next(iter(self.canonical_constraints.items()))
        new_params = ensure(torch.stack(params, dim=-1), constraint)
        return {key: new_params}

    @property
    def scale_factors(self):
        return [lambda w: torch.ones_like(w)] * self.size

    def params_from_data(self, x):
        new_x = to_one_hot(x, self.size)
        return probs_to_logits(new_x.sum(dim=0) / x.size(0)).unbind(dim=-1)

    def compute_hessian(self, data):
        raise NotImplementedError

    def __str__(self):
        return f'categorical({self.size})'


class OneHotCategorical(Categorical):
    def __init__(self, size, *, ensure_args=True):
        super(OneHotCategorical, self).__init__(size, ensure_args=ensure_args)
        self.size = size

    @property
    def dist(self) -> type(dist.Distribution):
        return dist.OneHotCategorical

    def compute_hessian(self, data):
        raise NotImplementedError

    def __str__(self):
        return f'one-hot-categorical({self.size})'


class GammaTrickCategorical(LikelihoodList):
    def __init__(self, size, *, ensure_args=True):
        super(GammaTrickCategorical, self).__init__(*[
            GammaTrick(Bernoulli(domain_size=1, ensure_args=ensure_args)) for _ in range(size)
        ], ensure_args=ensure_args)
        self.size = size

    @property
    def dist(self) -> type(dist.Distribution):
        if self.training:
            raise NotImplementedError
        return dist.OneHotCategorical

    def canonical_params(self, *params):
        if self.training:
            return super(GammaTrickCategorical, self).canonical_params(*params)

        self.train()  # To get the parameters from the gamma
        gamma_params = super(GammaTrickCategorical, self).canonical_params(*params)
        means = [super(GammaTrick, d).dist(**p).mean - 1 - d.noise.mean for d, p in zip(self.dist_list, gamma_params)]
        means = torch.stack(means, dim=-1) if params[0].numel() > 1 else torch.cat(means, dim=-1)
        discrete_value = ensure(means, constraints.simplex)
        self.eval()

        return {'probs': discrete_value}

    @property
    def canonical_constraints(self):
        if self.training:
            raise NotImplementedError
        return {'probs': constraints.simplex}

    def instantiate(self, *params):
        if self.training:
            return super(GammaTrickCategorical, self).instantiate(*params)

        params = sum(self.inverse_transform_params(*params), [])
        for d in self:
            d._counter = 1
        instance = self.dist(**self.canonical_params(*params))  # TODO BUG HERE (two ensure_params)
        for d in self:
            d._counter = 0

        return instance

    def compute_hessian(self, data):
        raise NotImplementedError

    def compute_lipschitz(self, data, original_hessian):
        raise NotImplementedError

    def __str__(self):
        return f'one-hot-categorical({len(self)})*'


class BernoulliTrickCategorical(LikelihoodList):
    def __init__(self, size, *, ensure_args=True):
        super(BernoulliTrickCategorical, self).__init__(*[
            Bernoulli(domain_size=1, ensure_args=ensure_args) for _ in range(size)
        ], ensure_args=ensure_args)

    @property
    def dist(self) -> type(dist.Distribution):
        if self.training:
            raise NotImplementedError
        return dist.OneHotCategorical

    def canonical_params(self, *params):
        if self.training:
            return super(BernoulliTrickCategorical, self).canonical_params(*params)

        means = super(BernoulliTrickCategorical, self).canonical_params(*params)
        means = [x['probs'] for x in means]
        means = torch.stack(means, dim=-1) if params[0].numel() > 1 else torch.cat(means, dim=-1)
        discrete_value = ensure(means, constraints.simplex)

        return {'probs': discrete_value}

    @property
    def canonical_constraints(self):
        if self.training:
            raise NotImplementedError
        return {'probs': constraints.simplex}

    def instantiate(self, *params):
        if self.training:
            return super(BernoulliTrickCategorical, self).instantiate(*params)

        params = sum(self.inverse_transform_params(*params), [])
        instance = self.dist(**self.canonical_params(*params))

        return instance

    def compute_hessian(self, data):
        raise NotImplementedError

    def compute_lipschitz(self, data, original_hessian):
        raise NotImplementedError

    def __str__(self):
        return f'one-hot-categorical({len(self)})+'

