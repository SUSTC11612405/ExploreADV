# mypy: allow-untyped-defs, no-strict-optional

from typing import Union, Optional, Tuple, Any
from typing_extensions import Literal
from abc import ABC
from abc import abstractmethod
import numpy as np
import eagerpy as ep
import logging
import warnings
from utils import flatten
from .base import Attack, LabelMixin


try:
    from numba.experimental import jitclass  # type: ignore
    import numba
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    # delay the error until the attack is initialized
    NUMBA_IMPORT_ERROR = e

    def jitclass(*args, **kwargs):
        def decorator(c):
            return c

        return decorator


else:
    NUMBA_IMPORT_ERROR = None

EPS = 1e-10


class Optimizer(object):  # pragma: no cover
    """ Base class for the trust-region optimization. If feasible, this optimizer solves the problem

        min_delta distance(x0, x + delta) s.t. ||delta||_2 <= r AND delta^T b = c AND min_ <= x + delta <= max_

        where x0 is the original sample, x is the current optimisation state, r is the trust-region radius,
        b is the current estimate of the normal vector of the decision boundary, c is the estimated distance of x
        to the trust region and [min_, max_] are the value constraints of the input. The function distance(.,.)
        is the distance measure to be optimised (e.g. L2, L1, L0).

    """

    def __init__(self):
        self.bfgsb = BFGSB()  # a box-constrained BFGS solver

    def solve(self, x0, x, b, _min, _max, c, r):
        x0, x, b = x0.astype(np.float64), x.astype(np.float64), b.astype(np.float64)
        _min, _max = _min.astype(np.float64), _max.astype(np.float64)
        cmax, cmaxnorm = self._max_logit_diff(x, b, _min, _max, c)

        if np.abs(cmax) < np.abs(c):
            # problem not solvable (boundary cannot be reached)
            if np.sqrt(cmaxnorm) < r:
                # make largest possible step towards boundary while staying within bounds
                _delta = self.optimize_boundary_s_t_trustregion(
                    x0, x, b, _min, _max, c, r
                )
            else:
                # make largest possible step towards boundary while staying within trust region
                _delta = self.optimize_boundary_s_t_trustregion(
                    x0, x, b, _min, _max, c, r
                )
        else:
            if cmaxnorm < r:
                # problem is solvable
                # proceed with standard optimization
                _delta = self.optimize_distance_s_t_boundary_and_trustregion(
                    x0, x, b, _min, _max, c, r
                )
            else:
                # problem might not be solvable
                bnorm = np.linalg.norm(b)
                minnorm = self._minimum_norm_to_boundary(x, b, _min, _max, c, bnorm)

                if minnorm <= r:
                    # problem is solvable, proceed with standard optimization
                    _delta = self.optimize_distance_s_t_boundary_and_trustregion(
                        x0, x, b, _min, _max, c, r
                    )
                else:
                    # problem not solvable (boundary cannot be reached)
                    # make largest step towards boundary within trust region
                    _delta = self.optimize_boundary_s_t_trustregion(
                        x0, x, b, _min, _max, c, r
                    )

        return _delta

    def _max_logit_diff(self, x, b, _ell, _u, c):
        """ Tests whether the (estimated) boundary can be reached within trust region. """
        N = x.shape[0]
        cmax = 0.0
        norm = 0.0

        if c > 0:
            for n in range(N):
                if b[n] > 0:
                    cmax += b[n] * (_u[n] - x[n])
                    norm += (_u[n] - x[n]) ** 2
                else:
                    cmax += b[n] * (_ell[n] - x[n])
                    norm += (x[n] - _ell[n]) ** 2
        else:
            for n in range(N):
                if b[n] > 0:
                    cmax += b[n] * (_ell[n] - x[n])
                    norm += (x[n] - _ell[n]) ** 2
                else:
                    cmax += b[n] * (_u[n] - x[n])
                    norm += (_u[n] - x[n]) ** 2

        return cmax, np.sqrt(norm)

    def _minimum_norm_to_boundary(self, x, b, _ell, _u, c, bnorm):
        """ Computes the minimum norm necessary to reach the boundary. More precisely, we aim to solve the
            following optimization problem

                min ||delta||_2^2 s.t. lower <= x + delta <= upper AND b.dot(delta) = c

            Lets forget about the box constraints for a second, i.e.

                min ||delta||_2^2 s.t. b.dot(delta) = c

            The dual of this problem is quite straight-forward to solve,

                g(lambda, delta) = ||delta||_2^2 + lambda * (c - b.dot(delta))

            The minimum of this Lagrangian is delta^* = lambda * b / 2, and so

                inf_delta g(lambda, delta) = lambda^2 / 4 ||b||_2^2 + lambda * c

            and so the optimal lambda, which maximizes inf_delta g(lambda, delta), is given by

                lambda^* = 2c / ||b||_2^2

            which in turn yields the optimal delta:

                delta^* = c * b / ||b||_2^2

            To take into account the box-constraints we perform a binary search over lambda and apply the box
            constraint in each step.
        """
        N = x.shape[0]

        lambda_lower = 2 * c / (bnorm ** 2 + EPS)
        lambda_upper = (
            np.sign(c) * np.inf
        )  # optimal initial point (if box-constraints are neglected)
        _lambda = lambda_lower
        k = 0

        # perform a binary search over lambda
        while True:
            # compute _c = b.dot([- _lambda * b / 2]_clip)
            k += 1
            _c = 0
            norm = 0

            if c > 0:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _u[n] - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
                    else:
                        max_step = _ell[n] - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
            else:
                for n in range(N):
                    lam_step = _lambda * b[n] / 2
                    if b[n] > 0:
                        max_step = _ell[n] - x[n]
                        delta_step = max(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2
                    else:
                        max_step = _u[n] - x[n]
                        delta_step = min(max_step, lam_step)
                        _c += b[n] * delta_step
                        norm += delta_step ** 2

            # adjust lambda
            if np.abs(_c) < np.abs(c):
                # increase absolute value of lambda
                if np.isinf(lambda_upper):
                    _lambda *= 2
                else:
                    lambda_lower = _lambda
                    _lambda = (lambda_upper - lambda_lower) / 2 + lambda_lower
            else:
                # decrease lambda
                lambda_upper = _lambda
                _lambda = (lambda_upper - lambda_lower) / 2 + lambda_lower

            # stopping condition
            if 0.999 * np.abs(c) - EPS < np.abs(_c) < 1.001 * np.abs(c) + EPS:
                break

        return np.sqrt(norm)

    def optimize_distance_s_t_boundary_and_trustregion(
        self, x0, x, b, _min, _max, c, r
    ):
        """ Find the solution to the optimization problem

            min_delta ||dx - delta||_p^p s.t. ||delta||_2^2 <= r^2 AND b^T delta = c AND min_ <= x + delta <= max_
        """
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])
        args = (x0, x, b, _min, _max, c, r)

        qk = self.bfgsb.solve(self.fun_and_jac, params0, bounds, args)
        return self._get_final_delta(
            qk[0], qk[1], x0, x, b, _min, _max, c, r, touchup=True
        )

    def optimize_boundary_s_t_trustregion_fun_and_jac(
        self, params, x0, x, b, _min, _max, c, r
    ):
        N = x0.shape[0]
        s = -np.sign(c)
        _mu = params[0]
        t = 1 / (2 * _mu + EPS)

        g = -_mu * r ** 2
        grad_mu = -(r ** 2)

        for n in range(N):
            d = -s * b[n] * t

            if d < _min[n] - x[n]:
                d = _min[n] - x[n]
            elif d > _max[n] - x[n]:
                d = _max[n] - x[n]
            else:
                grad_mu += (b[n] + 2 * _mu * d) * (b[n] / (2 * _mu ** 2 + EPS))

            grad_mu += d ** 2
            g += (b[n] + _mu * d) * d

        return -g, -np.array([grad_mu])

    def safe_div(self, nominator, denominator):
        if np.abs(denominator) > EPS:
            return nominator / denominator
        elif denominator >= 0:
            return nominator / EPS
        else:
            return -nominator / EPS

    def optimize_boundary_s_t_trustregion(self, x0, x, b, _min, _max, c, r):
        """ Find the solution to the optimization problem

            min_delta sign(c) b^T delta s.t. ||delta||_2^2 <= r^2 AND min_ <= x + delta <= max_

            Note: this optimization problem is independent of the Lp norm being optimized.

            Lagrangian: g(delta) = sign(c) b^T delta + mu * (||delta||_2^2 - r^2)
            Optimal delta: delta = - sign(c) * b / (2 * mu)
        """
        params0 = np.array([1.0])
        args = (x0, x, b, _min, _max, c, r)
        bounds = np.array([(0, np.inf)])

        qk = self.bfgsb.solve(
            self.optimize_boundary_s_t_trustregion_fun_and_jac, params0, bounds, args
        )

        _delta = self.safe_div(-b, 2 * qk[0])

        for n in range(x0.shape[0]):
            if _delta[n] < _min[n] - x[n]:
                _delta[n] = _min[n] - x[n]
            elif _delta[n] > _max[n] - x[n]:
                _delta[n] = _max[n] - x[n]

        return _delta


class BrendelBethgeAttack(Attack, LabelMixin):
    """Base class for the Brendel & Bethge adversarial attack [#Bren19]_, a powerful
    gradient-based adversarial attack that follows the adversarial boundary
    (the boundary between the space of adversarial and non-adversarial images as
    defined by the adversarial criterion) to find the minimum distance to the
    clean image.

    This is the reference implementation of the Brendel & Bethge attack.

    Implementation differs from the attack used in the paper in two ways:
    * The initial binary search is always using the full 10 steps (for ease of implementation).
    * The adaptation of the trust region over the course of optimisation is less
      greedy but is more robust, reliable and simpler (decay every K steps)

    Args:
        predict: forward pass function.
        overshoot : If 1 the attack tries to return exactly to the adversarial boundary
            in each iteration. For higher values the attack tries to overshoot
            over the boundary to ensure that the perturbed sample in each iteration
            is adversarial.
        steps : Maximum number of iterations to run. Might converge and stop
            before that.
        lr : Trust region radius, behaves similar to a learning rate. Smaller values
            decrease the step size in each iteration and ensure that the attack
            follows the boundary more faithfully.
        lr_decay : The trust region lr is multiplied with lr_decay in regular intervals (see
            lr_num_decay).
        lr_num_decay : Number of learning rate decays in regular intervals of
            length steps / lr_num_decay.
        momentum : Averaging of the boundary estimation over multiple steps. A momentum of
            zero would always take the current estimate while values closer to one
            average over a larger number of iterations.
        tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        binary_search_steps : Number of binary search steps used to find the adversarial boundary
            between the starting point and the clean image.
        targeted: if the attack is targeted.

    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer,
            Ivan Ustyuzhaninov, Matthias Bethge,
            "Accurate, reliable and fast robustness evaluation",
            33rd Conference on Neural Information Processing Systems (2019)
            https://arxiv.org/abs/1907.01003
    """

    def __init__(self, predict, loss_fn=None, overshoot: float = 1.1, steps: int = 1000,
                 lr: float = 0.1, lr_decay: float = 0.5, lr_num_decay: int = 20,
                 momentum: float = 0.8, binary_search_steps: int = 10,
                 clip_min=0., clip_max=1., targeted=False
    ):

        if NUMBA_IMPORT_ERROR is not None:
            raise NUMBA_IMPORT_ERROR  # pragma: no cover

        if "0.49." in numba.__version__:
            warnings.warn(
                "There are known issues with numba version 0.49 and we suggest using numba 0.50 or newer."
            )
        super(BrendelBethgeAttack, self).__init__(predict, loss_fn=loss_fn,
                                                 clip_min=clip_min,
                                                 clip_max=clip_max)
        self.predict = predict
        self.overshoot = overshoot
        self.steps = steps
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_num_decay = lr_num_decay
        self.momentum = momentum
        self.binary_search_steps = binary_search_steps
        self.targeted = targeted

        self._optimizer: Optimizer = self.instantiate_optimizer()

    def model(self, inputs):
        import torch
        from typing import cast

        def _model(x: torch.Tensor) -> torch.Tensor:
            with torch.set_grad_enabled(x.requires_grad):
                result = cast(torch.Tensor, self.predict(x))
            return result

        x, restore_type = ep.astensor_(inputs)
        z = ep.astensor(_model(x.raw))
        return restore_type(z)

    def get_is_adversarial(self, y):

        def is_adversarial(perturbed):
            outputs = self.model(perturbed)
            outputs_, restore_type = ep.astensor_(outputs)
            del perturbed, outputs
            classes = outputs_.argmax(axis=-1)
            if self.targeted:
                is_adv = classes == y
            else:
                is_adv = classes != y
            return restore_type(is_adv)

        return is_adversarial

    def perturb(self, x, starting_points, y=None, mask=None):

        """Applies the Brendel & Bethge attack.

        Parameters
        ----------
        x : Tensor that matches model type
            The original clean inputs.
        starting_points : Tensor of same type and shape as inputs
            Adversarial inputs to use as a starting points, in particular
            for targeted attacks.
        y : label tensor.
            - if None and self.targeted=False, compute y as predicted labels.
            - if self.targeted=True, then y must be the targeted labels.
        mask: mask tensor.
        """
        originals, classes = self._verify_and_process_inputs(x, y)
        originals, restore_type = ep.astensor_(originals)
        classes = ep.astensor(classes)

        is_adversarial = self.get_is_adversarial(classes)

        best_advs = ep.astensor(starting_points)
        assert is_adversarial(best_advs).all()

        # perform binary search to find adversarial boundary
        # TODO: Implement more efficient search with breaking condition
        N = len(originals)
        rows = range(N)

        if mask is None:
            min_, max_ = ep.full_like(originals, self.clip_min), ep.full_like(originals, self.clip_max)
        else:
            mask = ep.astensor(mask)
            # min_ = ep.where(mask > 0, 0, originals)
            # max_ = ep.where(mask > 0, 1, originals)
            min_ = ep.where(mask > 0, ep.maximum(originals - mask, self.clip_min), originals)
            max_ = ep.where(mask > 0, ep.minimum(originals + mask, self.clip_max), originals)

        min_np_flatten = min_.numpy().reshape((N, -1))
        max_np_flatten = max_.numpy().reshape((N, -1))

        x0 = originals
        x0_np_flatten = x0.numpy().reshape((N, -1))
        x1 = best_advs

        lower_bound = ep.zeros(x0, shape=(N,))
        upper_bound = ep.ones(x0, shape=(N,))

        for _ in range(self.binary_search_steps):
            epsilons = (lower_bound + upper_bound) / 2
            mid_points = self.mid_points(x0, x1, epsilons, min_, max_)
            is_advs = is_adversarial(mid_points)
            lower_bound = ep.where(is_advs, lower_bound, epsilons)
            upper_bound = ep.where(is_advs, epsilons, upper_bound)

        starting_points = self.mid_points(x0, x1, upper_bound, min_, max_)


        # function to compute logits_diff and gradient
        def loss_fun(x):
            logits = self.model(x)

            if self.targeted:
                c_minimize = best_other_classes(logits, classes)
                c_maximize = classes
            else:
                c_minimize = classes
                c_maximize = best_other_classes(logits, classes)

            logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
            assert logits_diffs.shape == (N,)

            return logits_diffs.sum(), logits_diffs

        value_and_grad = ep.value_and_grad_fn(x0, loss_fun, has_aux=True)

        def logits_diff_and_grads(x) -> Tuple[Any, Any]:
            _, logits_diffs, boundary = value_and_grad(x)
            return logits_diffs.numpy(), boundary.numpy().copy()

        x = starting_points
        lrs = self.lr * np.ones(N)
        lr_reduction_interval = max(1, int(self.steps / self.lr_num_decay))
        converged = np.zeros(N, dtype=np.bool)
        # counter = np.zeros(N, dtype=np.bool)
        rate_normalization = (max_np_flatten - min_np_flatten).sum(axis=-1)
        original_shape = x.shape
        _best_advs = best_advs.numpy()

        for step in range(1, self.steps + 1):
            if converged.all():
                break  # pragma: no cover

            # get logits and local boundary geometry
            # TODO: only perform forward pass on non-converged samples
            logits_diffs, _boundary = logits_diff_and_grads(x)

            # record optimal adversarials
            distances = self.norms(originals - x)
            source_norms = self.norms(originals - best_advs)

            closer = distances < source_norms
            is_advs = logits_diffs < 0
            closer = closer.logical_and(ep.from_numpy(x, is_advs))

            x_np_flatten = x.numpy().reshape((N, -1))

            if closer.any():
                _best_advs = best_advs.numpy().copy()
                _closer = closer.numpy().flatten()
                for idx in np.arange(N)[_closer]:
                    _best_advs[idx] = x_np_flatten[idx].reshape(original_shape[1:])

            # counter = np.where(closer.numpy().flatten(), 0, counter + 1)
            # converged = counter > 10
            # print(step, converged, counter)
            # 100 steps: [0.08128784 0.04055694 0.05031109 0.0882149  0.0354948]

            best_advs = ep.from_numpy(x, _best_advs)

            # denoise estimate of boundary using a short history of the boundary
            if step == 1:
                boundary = _boundary
            else:
                boundary = (1 - self.momentum) * _boundary + self.momentum * boundary

            # learning rate adaptation
            if (step + 1) % lr_reduction_interval == 0:
                lrs *= self.lr_decay

            # compute optimal step within trust region depending on metric
            x = x.reshape((N, -1))
            region = lrs * rate_normalization

            # we aim to slight overshoot over the boundary to stay within the adversarial region
            corr_logits_diffs = np.where(
                -logits_diffs < 0,
                -self.overshoot * logits_diffs,
                -(2 - self.overshoot) * logits_diffs,
            )

            # employ solver to find optimal step within trust region
            # for each sample
            deltas, k = [], 0

            for sample in range(N):
                if converged[sample]:
                    # don't perform optimisation on converged samples
                    deltas.append(
                        np.zeros_like(x0_np_flatten[sample])
                    )  # pragma: no cover
                else:
                    _x0 = x0_np_flatten[sample]
                    _x = x_np_flatten[sample]
                    _b = boundary[k].flatten()
                    _min = min_np_flatten[sample]
                    _max = max_np_flatten[sample]
                    _c = corr_logits_diffs[k]
                    r = region[sample]

                    delta = self._optimizer.solve(  # type: ignore
                        _x0, _x, _b, _min, _max, _c, r
                    )
                    deltas.append(delta)

                    k += 1  # idx of masked sample

            deltas = np.stack(deltas)
            deltas = ep.from_numpy(x, deltas.astype(np.float32))  # type: ignore

            # add step to current perturbation
            x = (x + ep.astensor(deltas)).reshape(original_shape)

        return restore_type(best_advs)

    @abstractmethod
    def instantiate_optimizer(self) -> Optimizer:
        raise NotImplementedError

    @abstractmethod
    def norms(self, x: ep.Tensor) -> ep.Tensor:
        raise NotImplementedError

    @abstractmethod
    def mid_points(
        self,
        x0: ep.Tensor,
        x1: ep.Tensor,
        epsilons: ep.Tensor,
        min_,
        max_
    ) -> ep.Tensor:
        raise NotImplementedError


def best_other_classes(logits: ep.Tensor, exclude: ep.Tensor) -> ep.Tensor:
    other_logits = logits - ep.onehot_like(logits, exclude, value=np.inf)
    return other_logits.argmax(axis=-1)


class LinfinityBrendelBethgeAttack(BrendelBethgeAttack):
    """L-infinity variant of the Brendel & Bethge adversarial attack. [#Bren19]_
    This is a powerful gradient-based adversarial attack that follows the
    adversarial boundary (the boundary between the space of adversarial and
    non-adversarial images as defined by the adversarial criterion) to find
    the minimum distance to the clean image.

    This is the reference implementation of the Brendel & Bethge attack.

    References:
        .. [#Bren19] Wieland Brendel, Jonas Rauber, Matthias Kümmerer,
           Ivan Ustyuzhaninov, Matthias Bethge,
           "Accurate, reliable and fast robustness evaluation",
           33rd Conference on Neural Information Processing Systems (2019)
           https://arxiv.org/abs/1907.01003
   """

    def instantiate_optimizer(self):
        return LinfOptimizer()

    def norms(self, x: ep.Tensor) -> ep.Tensor:
        return flatten(x).norms.linf(axis=-1)

    def mid_points(
        self,
        x0: ep.Tensor,
        x1: ep.Tensor,
        epsilons: ep.Tensor,
        min_,
        max_
    ):
        # returns a point between x0 and x1 where
        # epsilon = 0 returns x0 and epsilon = 1
        delta = x1 - x0
        s = max_ - min_
        # get epsilons in right shape for broadcasting
        epsilons = epsilons.reshape(epsilons.shape + (1,) * (x0.ndim - 1))

        clipped_delta = ep.where(delta < -epsilons * s, -epsilons * s, delta)
        clipped_delta = ep.where(
            clipped_delta > epsilons * s, epsilons * s, clipped_delta
        )
        return x0 + clipped_delta


@jitclass(spec=[])
class BFGSB(object):
    def __init__(self):
        pass

    def solve(
        self, fun_and_jac, q0, bounds, args, ftol=1e-10, pgtol=-1e-5, maxiter=None
    ):
        N = q0.shape[0]

        if maxiter is None:
            maxiter = N * 200

        l = bounds[:, 0]  # noqa: E741
        u = bounds[:, 1]

        func_calls = 0

        old_fval, gfk = fun_and_jac(q0, *args)
        func_calls += 1

        k = 0
        Hk = np.eye(N)

        # Sets the initial step guess to dx ~ 1
        qk = q0
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        # gnorm = np.amax(np.abs(gfk))
        _gfk = gfk

        # Compare with implementation BFGS-B implementation
        # in https://github.com/andrewhooker/PopED/blob/master/R/bfgsb_min.R

        while k < maxiter:
            # check if projected gradient is still large enough
            pg_norm = 0
            for v in range(N):
                if _gfk[v] < 0:
                    gv = max(qk[v] - u[v], _gfk[v])
                else:
                    gv = min(qk[v] - l[v], _gfk[v])

                if pg_norm < np.abs(gv):
                    pg_norm = np.abs(gv)

            if pg_norm < pgtol:
                break

            # get cauchy point
            x_cp = self._cauchy_point(qk, l, u, _gfk.copy(), Hk)
            qk1 = self._subspace_min(qk, l, u, x_cp, _gfk.copy(), Hk)
            pk = qk1 - qk

            (
                alpha_k,
                fc,
                gc,
                old_fval,
                old_old_fval,
                gfkp1,
                fnev,
            ) = self._line_search_wolfe(
                fun_and_jac, qk, pk, _gfk, old_fval, old_old_fval, l, u, args
            )
            func_calls += fnev

            if alpha_k is None:
                break

            if np.abs(old_fval - old_old_fval) <= (ftol + ftol * np.abs(old_fval)):
                break

            qkp1 = self._project(qk + alpha_k * pk, l, u)

            if gfkp1 is None:
                _, gfkp1 = fun_and_jac(qkp1, *args)

            sk = qkp1 - qk
            qk = qkp1

            yk = np.zeros_like(qk)
            for k3 in range(N):
                yk[k3] = gfkp1[k3] - _gfk[k3]

                if np.abs(yk[k3]) < 1e-4:
                    yk[k3] = -1e-4

            _gfk = gfkp1

            k += 1

            # update inverse Hessian matrix
            Hk_sk = Hk.dot(sk)

            sk_yk = 0
            sk_Hk_sk = 0
            for v in range(N):
                sk_yk += sk[v] * yk[v]
                sk_Hk_sk += sk[v] * Hk_sk[v]

            if np.abs(sk_yk) >= 1e-8:
                rhok = 1.0 / sk_yk
            else:
                rhok = 100000.0

            if np.abs(sk_Hk_sk) >= 1e-8:
                rsk_Hk_sk = 1.0 / sk_Hk_sk
            else:
                rsk_Hk_sk = 100000.0

            for v in range(N):
                for w in range(N):
                    Hk[v, w] += yk[v] * yk[w] * rhok - Hk_sk[v] * Hk_sk[w] * rsk_Hk_sk

        return qk

    def _cauchy_point(self, x, l, u, g, B):  # noqa: E741
        # finds the cauchy point for q(x)=x'Gx+x'd s$t. l<=x<=u
        # g=G*x+d #gradient of q(x)
        # converted from r-code: https://github.com/andrewhooker/PopED/blob/master/R/cauchy_point.R
        n = x.shape[0]
        t = np.zeros_like(x)
        d = np.zeros_like(x)

        for i in range(n):
            if g[i] < 0:
                t[i] = (x[i] - u[i]) / (g[i] - EPS)
            elif g[i] > 0:
                t[i] = (x[i] - l[i]) / (g[i] + EPS)
            elif g[i] == 0:
                t[i] = np.inf

            if t[i] == 0:
                d[i] = 0
            else:
                d[i] = -g[i]

        ts = t.copy()
        ts = ts[ts != 0]
        ts = np.sort(ts)

        df = g.dot(d)
        d2f = d.dot(B.dot(d))

        if d2f < 1e-10:
            return x

        dt_min = -df / d2f
        t_old = 0
        i = 0
        z = np.zeros_like(x)

        while i < ts.shape[0] and dt_min >= (ts[i] - t_old):
            ind = ts[i] < t
            d[~ind] = 0
            z = z + (ts[i] - t_old) * d
            df = g.dot(d) + d.dot(B.dot(z))
            d2f = d.dot(B.dot(d))
            dt_min = df / (d2f + 1e-8)
            t_old = ts[i]
            i += 1

        dt_min = max(dt_min, 0)
        t_old = t_old + dt_min
        x_cp = x - t_old * g
        temp = x - t * g
        x_cp[t_old > t] = temp[t_old > t]

        return x_cp

    def _subspace_min(self, x, l, u, x_cp, d, G):  # noqa: E741
        # converted from r-code: https://github.com/andrewhooker/PopED/blob/master/R/subspace_min.R
        n = x.shape[0]
        Z = np.eye(n)
        fixed = (x_cp <= l + 1e-8) + (x_cp >= u - 1e8)

        if np.all(fixed):
            x = x_cp
            return x

        Z = Z[:, ~fixed]
        rgc = Z.T.dot(d + G.dot(x_cp - x))
        rB = Z.T.dot(G.dot(Z)) + 1e-10 * np.eye(Z.shape[1])
        d[~fixed] = np.linalg.solve(rB, rgc)
        d[~fixed] = -d[~fixed]
        alpha = 1
        temp1 = alpha

        for i in np.arange(n)[~fixed]:
            dk = d[i]
            if dk < 0:
                temp2 = l[i] - x_cp[i]
                if temp2 >= 0:
                    temp1 = 0
                else:
                    if dk * alpha < temp2:
                        temp1 = temp2 / (dk - EPS)
                    else:
                        temp2 = u[i] - x_cp[i]
            else:
                temp2 = u[i] - x_cp[i]
                if temp1 <= 0:
                    temp1 = 0
                else:
                    if dk * alpha > temp2:
                        temp1 = temp2 / (dk + EPS)

            alpha = min(temp1, alpha)

        return x_cp + alpha * Z.dot(d[~fixed])

    def _project(self, q, l, u):  # noqa: E741
        N = q.shape[0]
        for k in range(N):
            if q[k] < l[k]:
                q[k] = l[k]
            elif q[k] > u[k]:
                q[k] = u[k]

        return q

    def _line_search_armijo(
        self,
        fun_and_jac,
        pt,
        dpt,
        func_calls,
        m,
        gk,
        l,  # noqa: E741
        u,
        x0,
        x,
        b,
        min_,
        max_,
        c,
        r,
    ):
        ls_rho = 0.6
        ls_c = 1e-4
        ls_alpha = 1

        t = m * ls_c

        for k2 in range(100):
            ls_pt = self._project(pt + ls_alpha * dpt, l, u)

            gkp1, dgkp1 = fun_and_jac(ls_pt, x0, x, b, min_, max_, c, r)
            func_calls += 1

            if gk - gkp1 >= ls_alpha * t:
                break
            else:
                ls_alpha *= ls_rho

        return ls_alpha, ls_pt, gkp1, dgkp1, func_calls

    def _line_search_wolfe(  # noqa: C901
        self,
        fun_and_jac,
        xk,
        pk,
        gfk,
        old_fval,
        old_old_fval,
        l,  # noqa: #E741
        u,
        args,
    ):
        """Find alpha that satisfies strong Wolfe conditions.
        Uses the line search algorithm to enforce strong Wolfe conditions
        Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-60
        For the zoom phase it uses an algorithm by
        Outputs: (alpha0, gc, fc)
        """
        c1 = 1e-4
        c2 = 0.9
        N = xk.shape[0]
        _ls_fc = 0
        _ls_ingfk = None

        alpha0 = 0
        phi0 = old_fval

        derphi0 = 0
        for v in range(N):
            derphi0 += gfk[v] * pk[v]

        if derphi0 == 0:
            derphi0 = 1e-8
        elif np.abs(derphi0) < 1e-8:
            derphi0 = np.sign(derphi0) * 1e-8

        alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_old_fval) / derphi0)

        if alpha1 == 0:
            # This shouldn't happen. Perhaps the increment has slipped below
            # machine precision?  For now, set the return variables skip the
            # useless while loop, and raise warnflag=2 due to possible imprecision.
            # print("Slipped below machine precision.")
            alpha_star = None
            fval_star = old_fval
            old_fval = old_old_fval
            fprime_star = None

        _xkp1 = self._project(xk + alpha1 * pk, l, u)
        phi_a1, _ls_ingfk = fun_and_jac(_xkp1, *args)
        _ls_fc += 1
        # derphi_a1 = phiprime(alpha1)  evaluated below

        phi_a0 = phi0
        derphi_a0 = derphi0

        i = 1
        maxiter = 10
        while 1:  # bracketing phase
            # print("   (ls) in while loop: ", alpha1, alpha0)
            if alpha1 == 0:
                break
            if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or (
                (phi_a1 >= phi_a0) and (i > 1)
            ):
                # inlining zoom for performance reasons
                #                 alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi0, derphi0, pk, xk
                # zoom signature: (a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi0, derphi0, pk, xk)
                # INLINE START
                k = 0
                delta1 = 0.2  # cubic interpolant check
                delta2 = 0.1  # quadratic interpolant check
                phi_rec = phi0
                a_rec = 0
                a_hi = alpha1
                a_lo = alpha0
                phi_lo = phi_a0
                phi_hi = phi_a1
                derphi_lo = derphi_a0
                while 1:
                    # interpolate to find a trial step length between a_lo and a_hi
                    # Need to choose interpolation here.  Use cubic interpolation and then if the
                    #  result is within delta * dalpha or outside of the interval bounded by a_lo or a_hi
                    #  then use quadratic interpolation, if the result is still too close, then use bisection

                    dalpha = a_hi - a_lo
                    if dalpha < 0:
                        a, b = a_hi, a_lo
                    else:
                        a, b = a_lo, a_hi

                    # minimizer of cubic interpolant
                    #    (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
                    #      if the result is too close to the end points (or out of the interval)
                    #         then use quadratic interpolation with phi_lo, derphi_lo and phi_hi
                    #      if the result is stil too close to the end points (or out of the interval)
                    #         then use bisection

                    if k > 0:
                        cchk = delta1 * dalpha
                        a_j = self._cubicmin(
                            a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                        )
                    if (
                        (k == 0)
                        or (a_j is None)
                        or (a_j > b - cchk)
                        or (a_j < a + cchk)
                    ):
                        qchk = delta2 * dalpha
                        a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                        if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                            a_j = a_lo + 0.5 * dalpha

                    # Check new value of a_j
                    _xkp1 = self._project(xk + a_j * pk, l, u)
                    # if _xkp1[1] < 0:
                    #     _xkp1[1] = 0
                    phi_aj, _ls_ingfk = fun_and_jac(_xkp1, *args)

                    derphi_aj = 0
                    for v in range(N):
                        derphi_aj += _ls_ingfk[v] * pk[v]

                    if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_j
                        phi_hi = phi_aj
                    else:
                        if abs(derphi_aj) <= -c2 * derphi0:
                            a_star = a_j
                            val_star = phi_aj
                            valprime_star = _ls_ingfk
                            break
                        if derphi_aj * (a_hi - a_lo) >= 0:
                            phi_rec = phi_hi
                            a_rec = a_hi
                            a_hi = a_lo
                            phi_hi = phi_lo
                        else:
                            phi_rec = phi_lo
                            a_rec = a_lo
                        a_lo = a_j
                        phi_lo = phi_aj
                        derphi_lo = derphi_aj
                    k += 1
                    if k > maxiter:
                        a_star = a_j
                        val_star = phi_aj
                        valprime_star = None
                        break

                alpha_star = a_star
                fval_star = val_star
                fprime_star = valprime_star
                fnev = k
                ## INLINE END

                _ls_fc += fnev
                break

            i += 1
            if i > maxiter:
                break

            _xkp1 = self._project(xk + alpha1 * pk, l, u)
            _, _ls_ingfk = fun_and_jac(_xkp1, *args)
            derphi_a1 = 0
            for v in range(N):
                derphi_a1 += _ls_ingfk[v] * pk[v]
            _ls_fc += 1
            if abs(derphi_a1) <= -c2 * derphi0:
                alpha_star = alpha1
                fval_star = phi_a1
                fprime_star = _ls_ingfk
                break

            if derphi_a1 >= 0:
                # alpha_star, fval_star, fprime_star, fnev, _ls_ingfk = _zoom(
                #     alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi0, derphi0, pk, xk
                # )
                #
                # INLINE START
                maxiter = 10
                k = 0
                delta1 = 0.2  # cubic interpolant check
                delta2 = 0.1  # quadratic interpolant check
                phi_rec = phi0
                a_rec = 0
                a_hi = alpha0
                a_lo = alpha1
                phi_lo = phi_a1
                phi_hi = phi_a0
                derphi_lo = derphi_a1
                while 1:
                    # interpolate to find a trial step length between a_lo and a_hi
                    # Need to choose interpolation here.  Use cubic interpolation and then if the
                    #  result is within delta * dalpha or outside of the interval bounded by a_lo or a_hi
                    #  then use quadratic interpolation, if the result is still too close, then use bisection

                    dalpha = a_hi - a_lo
                    if dalpha < 0:
                        a, b = a_hi, a_lo
                    else:
                        a, b = a_lo, a_hi

                    # minimizer of cubic interpolant
                    #    (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
                    #      if the result is too close to the end points (or out of the interval)
                    #         then use quadratic interpolation with phi_lo, derphi_lo and phi_hi
                    #      if the result is stil too close to the end points (or out of the interval)
                    #         then use bisection

                    if k > 0:
                        cchk = delta1 * dalpha
                        a_j = self._cubicmin(
                            a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                        )
                    if (
                        (k == 0)
                        or (a_j is None)
                        or (a_j > b - cchk)
                        or (a_j < a + cchk)
                    ):
                        qchk = delta2 * dalpha
                        a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                        if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                            a_j = a_lo + 0.5 * dalpha

                    # Check new value of a_j
                    _xkp1 = self._project(xk + a_j * pk, l, u)
                    phi_aj, _ls_ingfk = fun_and_jac(_xkp1, *args)
                    derphi_aj = 0
                    for v in range(N):
                        derphi_aj += _ls_ingfk[v] * pk[v]
                    if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_j
                        phi_hi = phi_aj
                    else:
                        if abs(derphi_aj) <= -c2 * derphi0:
                            a_star = a_j
                            val_star = phi_aj
                            valprime_star = _ls_ingfk
                            break
                        if derphi_aj * (a_hi - a_lo) >= 0:
                            phi_rec = phi_hi
                            a_rec = a_hi
                            a_hi = a_lo
                            phi_hi = phi_lo
                        else:
                            phi_rec = phi_lo
                            a_rec = a_lo
                        a_lo = a_j
                        phi_lo = phi_aj
                        derphi_lo = derphi_aj
                    k += 1
                    if k > maxiter:
                        a_star = a_j
                        val_star = phi_aj
                        valprime_star = None
                        break

                alpha_star = a_star
                fval_star = val_star
                fprime_star = valprime_star
                fnev = k
                ## INLINE END

                _ls_fc += fnev
                break

            alpha2 = 2 * alpha1  # increase by factor of two on each iteration
            i = i + 1
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            _xkp1 = self._project(xk + alpha1 * pk, l, u)
            phi_a1, _ls_ingfk = fun_and_jac(_xkp1, *args)
            _ls_fc += 1
            derphi_a0 = derphi_a1

            # stopping test if lower function not found
            if i > maxiter:
                alpha_star = alpha1
                fval_star = phi_a1
                fprime_star = None
                break

        return alpha_star, _ls_fc, _ls_fc, fval_star, old_fval, fprime_star, _ls_fc

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        # finds the minimizer for a cubic polynomial that goes through the
        #  points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
        #
        # if no minimizer can be found return None
        #
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        C = fpa
        db = b - a
        dc = c - a
        if (db == 0) or (dc == 0) or (b == c):
            return None
        denom = (db * dc) ** 2 * (db - dc)
        A = dc ** 2 * (fb - fa - C * db) - db ** 2 * (fc - fa - C * dc)
        B = -(dc ** 3) * (fb - fa - C * db) + db ** 3 * (fc - fa - C * dc)

        A /= denom
        B /= denom
        radical = B * B - 3 * A * C
        if radical < 0:
            return None
        if A == 0:
            return None
        xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        return xmin

    def _quadmin(self, a, fa, fpa, b, fb):
        # finds the minimizer for a quadratic polynomial that goes through
        #  the points (a,fa), (b,fb) with derivative at a of fpa
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        D = fa
        C = fpa
        db = b - a * 1.0
        if db == 0:
            return None
        B = (fb - D - C * db) / (db * db)
        if B <= 0:
            return None
        xmin = a - C / (2.0 * B)
        return xmin


if NUMBA_IMPORT_ERROR is None:
    spec = [("bfgsb", BFGSB.class_type.instance_type)]  # type: ignore
else:
    spec = []  # pragma: no cover


@jitclass(spec=spec)
class LinfOptimizer(Optimizer):
    def optimize_distance_s_t_boundary_and_trustregion(
        self, x0, x, b, _min, _max, c, r
    ):
        """ Find the solution to the optimization problem

            min_delta ||dx - delta||_p^p s.t. ||delta||_2^2 <= r^2 AND b^T delta = c AND min_ <= x + delta <= max_
        """
        params0 = np.array([0.0, 0.0])
        bounds = np.array([(-np.inf, np.inf), (0, np.inf)])

        return self.binary_search(params0, bounds, x0, x, b, _min, _max, c, r)

    def binary_search(
        self, q0, bounds, x0, x, b, _min, _max, c, r, etol=1e-6, maxiter=1000
    ):
        # perform binary search over epsilon
        epsilon = np.max(_max - _min) / 2.0
        eps_low = 0.0
        eps_high = np.max(_max - _min)
        func_calls = 0

        bnorm = np.linalg.norm(b)
        lambda0 = 2 * c / (bnorm ** 2 + EPS)

        k = 0

        while eps_high - eps_low > etol:
            fun, nfev, _lambda0 = self.fun(
                epsilon, x0, x, b, _min, _max, c, r, lambda0=lambda0
            )
            func_calls += nfev
            if fun > -np.inf:
                # decrease epsilon
                eps_high = epsilon
                lambda0 = _lambda0
            else:
                # increase epsilon
                eps_low = epsilon

            k += 1
            epsilon = (eps_high - eps_low) / 2.0 + eps_low

            if k > 20:
                break

        delta = self._get_final_delta(
            lambda0, eps_high, x0, x, b, _min, _max, c, r, touchup=True
        )
        return delta

    def _Linf_bounds(self, x0, epsilon, ell, u):
        N = x0.shape[0]
        _ell = np.empty_like(x0)
        _u = np.empty_like(x0)
        for i in range(N):
            nx, px = x0[i] - epsilon, x0[i] + epsilon
            if nx > ell[i]:
                _ell[i] = nx
            else:
                _ell[i] = ell[i]

            if px < u[i]:
                _u[i] = px
            else:
                _u[i] = u[i]

        return _ell, _u

    def fun(self, epsilon, x0, x, b, ell, u, c, r, lambda0=None):
        """ Computes the minimum norm necessary to reach the boundary. More precisely, we aim to solve the
            following optimization problem

                min ||delta||_2^2 s.t. lower <= x + delta <= upper AND b.dot(delta) = c

            Lets forget about the box constraints for a second, i.e.

                min ||delta||_2^2 s.t. b.dot(delta) = c

            The dual of this problem is quite straight-forward to solve,

                g(lambda, delta) = ||delta||_2^2 + lambda * (c - b.dot(delta))

            The minimum of this Lagrangian is delta^* = lambda * b / 2, and so

                inf_delta g(lambda, delta) = lambda^2 / 4 ||b||_2^2 + lambda * c

            and so the optimal lambda, which maximizes inf_delta g(lambda, delta), is given by

                lambda^* = 2c / ||b||_2^2

            which in turn yields the optimal delta:

                delta^* = c * b / ||b||_2^2

            To take into account the box-constraints we perform a binary search over lambda and apply the box
            constraint in each step.
        """
        N = x.shape[0]

        # new box constraints
        _ell, _u = self._Linf_bounds(x0, epsilon, ell, u)

        # initialize lambda
        _lambda = lambda0

        # compute delta and determine active set
        k = 0

        lambda_max, lambda_min = 1e10, -1e10

        # check whether problem is actually solvable (i.e. check whether boundary constraint can be reached)
        max_c = 0
        min_c = 0

        for n in range(N):
            if b[n] > 0:
                max_c += b[n] * (_u[n] - x[n])
                min_c += b[n] * (_ell[n] - x[n])
            else:
                max_c += b[n] * (_ell[n] - x[n])
                min_c += b[n] * (_u[n] - x[n])

        if c > max_c or c < min_c:
            return -np.inf, k, _lambda

        while True:
            k += 1
            _c = 0
            norm = 0
            _active_bnorm = 0

            for n in range(N):
                lam_step = _lambda * b[n] / 2
                if lam_step + x[n] < _ell[n]:
                    delta_step = _ell[n] - x[n]
                elif lam_step + x[n] > _u[n]:
                    delta_step = _u[n] - x[n]
                else:
                    delta_step = lam_step
                    _active_bnorm += b[n] ** 2

                _c += b[n] * delta_step
                norm += delta_step ** 2

            if 0.9999 * np.abs(c) - EPS < np.abs(_c) < 1.0001 * np.abs(c) + EPS:
                if norm > r ** 2:
                    return -np.inf, k, _lambda
                else:
                    return -epsilon, k, _lambda
            else:
                # update lambda according to active variables
                if _c > c:
                    lambda_max = _lambda
                else:
                    lambda_min = _lambda
                #
                if _active_bnorm == 0:
                    # update is stepping out of feasible region, fallback to binary search
                    _lambda = (lambda_max - lambda_min) / 2 + lambda_min
                else:
                    _lambda += 2 * (c - _c) / (_active_bnorm + EPS)

                dlambda = lambda_max - lambda_min
                if (
                    _lambda > lambda_max - 0.1 * dlambda
                    or _lambda < lambda_min + 0.1 * dlambda
                ):
                    # update is stepping out of feasible region, fallback to binary search
                    _lambda = (lambda_max - lambda_min) / 2 + lambda_min

    def _get_final_delta(self, lam, eps, x0, x, b, min_, max_, c, r, touchup=True):
        N = x.shape[0]
        delta = np.empty_like(x0)

        # new box constraints
        _ell, _u = self._Linf_bounds(x0, eps, min_, max_)

        for n in range(N):
            lam_step = lam * b[n] / 2
            if lam_step + x[n] < _ell[n]:
                delta[n] = _ell[n] - x[n]
            elif lam_step + x[n] > _u[n]:
                delta[n] = _u[n] - x[n]
            else:
                delta[n] = lam_step

        return delta

    def _distance(self, x0, x):
        return np.abs(x0 - x).max()
