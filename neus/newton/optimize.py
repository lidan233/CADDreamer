
from __future__ import annotations

import warnings
import concurrent.futures
import functools
import itertools
import numpy as np
from scipy.optimize import minimize, Bounds
import time
from typing import Any, Callable
from numpy.typing import ArrayLike


__all__ = [
    "minimize_parallel",
]


class EvalParallel:
    def __init__(
        self,
        fun: Callable,
        jac: Any | None = None,
        args: tuple[Any] = (),
        eps: float = 1e-8,
        executor=concurrent.futures.ProcessPoolExecutor(),
        forward: bool = True,
        loginfo: bool = False,
        verbose: bool = False,
        n: int = 1,
    ):
        self.fun_in = fun
        self.jac_in = jac
        self.eps = eps
        self.forward = forward
        self.loginfo = loginfo
        self.verbose = verbose
        self.x_val = None
        self.fun_val = None
        self.jac_val = None
        if not (isinstance(args, list) or isinstance(args, tuple)):
            self.args = (args,)
        else:
            self.args = tuple(args)
        self.n = n
        self.executor = executor
        if self.loginfo:
            self.info = {k: [] for k in ["x", "fun", "jac"]}
        self.np_precision = np.finfo(float).eps

    # static helper methods are used for parallel execution with map()
    @staticmethod
    def _eval_approx_args(
        args: tuple[Any], eps_at: float, fun: Any, x: ArrayLike, eps: float
    ):
        # 'fun' has additional 'args'
        if eps_at == 0:
            x_ = x
        elif eps_at <= len(x):
            x_ = x.copy()
            x_[eps_at - 1] += eps
        else:
            x_ = x.copy()
            x_[eps_at - 1 - len(x)] -= eps
        return fun(x_, *args)

    @staticmethod
    def _eval_approx(eps_at: float, fun: Callable, x: ArrayLike, eps: float):
        # 'fun' has no additional 'args'
        if eps_at == 0:
            x_ = x
        elif eps_at <= len(x):
            x_ = x.copy()
            x_[eps_at - 1] += eps
        else:
            x_ = x.copy()
            x_[eps_at - 1 - len(x)] -= eps
        return fun(x_)

    @staticmethod
    def _eval_fun_jac_args(
        args: tuple[Any],
        which: int,
        fun: Callable,
        jac: Callable,
        x: ArrayLike,
    ):
        # 'fun' and 'jec; have additional 'args'
        if which == 0:
            return fun(x, *args)
        return np.array(jac(x, *args))

    @staticmethod
    def _eval_fun_jac(which: int, fun: Callable, jac: Callable, x: ArrayLike):
        # 'fun' and 'jac' have no additionals 'args'
        if which == 0:
            return fun(x)
        return np.array(jac(x))

    def eval_parallel(self, x: ArrayLike):
        # function to evaluate 'fun' and 'jac' in parallel
        # - if 'jac' is None, the gradient is computed numerically
        # - if 'forward' is True, the numerical gradient uses the
        #       forward difference method,
        #       otherwise, the central difference method is used
        x = np.array(x)
        if self.x_val is not None and all(abs(self.x_val - x) <= self.np_precision * 2):
            if self.verbose:
                print("re-use")
        else:
            self.x_val = x.copy()
            if self.jac_in is None:
                if self.forward:
                    eps_at = range(len(x) + 1)
                else:
                    eps_at = range(2 * len(x) + 1)

                # pack 'self.args' into function because otherwise it
                # cannot be serialized by
                # 'concurrent.futures.ProcessPoolExecutor()'
                if len(self.args) > 0:
                    ftmp = functools.partial(self._eval_approx_args, self.args)
                else:
                    ftmp = self._eval_approx

                ret = self.executor.map(
                    ftmp,
                    eps_at,
                    itertools.repeat(self.fun_in),
                    itertools.repeat(x),
                    itertools.repeat(self.eps),
                )
                ret = np.array(list(ret))
                self.fun_val = ret[0]
                if self.forward:
                    self.jac_val = (ret[1 : (len(x) + 1)] - self.fun_val) / self.eps
                else:
                    self.jac_val = (
                        ret[1 : (len(x) + 1)] - ret[(len(x) + 1) : 2 * len(x) + 1]
                    ) / (2 * self.eps)

            # 'jac' function is not None
            else:
                if len(self.args) > 0:
                    ftmp = functools.partial(self._eval_fun_jac_args, self.args)
                else:
                    ftmp = self._eval_fun_jac

                ret = self.executor.map(
                    ftmp,
                    [0, 1],
                    itertools.repeat(self.fun_in),
                    itertools.repeat(self.jac_in),
                    itertools.repeat(x),
                )
                ret = list(ret)
                self.fun_val = ret[0]
                self.jac_val = ret[1]

            self.jac_val = self.jac_val.reshape((self.n,))

            if self.loginfo:
                self.info["fun"].append(self.fun_val)
                if self.n >= 2:
                    self.info["x"].append(self.x_val.tolist())
                    self.info["jac"].append(self.jac_val.tolist())
                else:
                    self.info["x"].append(self.x_val[0])
                    self.info["jac"].append(self.jac_val[0])
        return None

    def fun(self, x: ArrayLike):
        self.eval_parallel(x=x)
        if self.verbose:
            print("fun(" + str(x) + ") = " + str(self.fun_val))
        return self.fun_val

    def jac(self, x):
        self.eval_parallel(x=x)
        if self.verbose:
            print("jac(" + str(x) + ") = " + str(self.jac_val))
        return self.jac_val



from scipy.optimize._numdiff import approx_derivative
def _convert_jac(params, constraints, new_bounds):
    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function. The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun):
                def cjac(x, *args):
                    from scipy.optimize._optimize import _check_clip_x
                    x = _check_clip_x(x, new_bounds)
                    return approx_derivative(fun, x, method='2-point',
                                                 abs_step=1e-8, args=args,
                                                 bounds=new_bounds)

                return cjac

            cjac = cjac_factory(con['fun'])

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())},)

    a_eq = np.vstack([con['jac'](params, *con['args']) for con in cons['eq']])
    import sympy
    _, inds = sympy.Matrix(a_eq).T.rref()

    return [constraints[ind] for ind in inds]


def minimize_parallel(
    fun: Callable,
    x0: ArrayLike,
    args: tuple[Any] = (),
    jac: Callable | None = None,
    bounds: Bounds | None = None,
    tol: float | None = None,
    options: dict | None = None,
    callback: Callable | None = None,
    parallel: dict | None = None,
    constraints: tuple | None= None
):
    try:
        n = len(x0)
    except Exception:
        n = 1

    if jac is True:
        raise ValueError(
            "'fun' returning the function AND its "
            "gradient is not supported.\n"
            "Please specify separate functions in "
            "'fun' and 'jac'."
        )

    # update default options with specified options
    options_used = {
        "disp": None,
        "maxcor": 10,
        "ftol": 2.220446049250313e-09,
        "gtol": 1e-05,
        "eps": 1e-08,
        "maxfun": 15000,
        "maxiter": 15000,
        "iprint": -1,
        "maxls": 20,
    }
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError("argument 'options' must be of type 'dict'")
        options_used.update(options)
    if tol is not None:
        if options is not None and "gtol" in options:
            warnings.warn(
                "'tol' is ignored and 'gtol' in 'options' is used instead.",
                RuntimeWarning,
            )
        else:
            options_used["gtol"] = tol

    parallel_used = {
        "max_workers": None,
        "forward": True,
        "verbose": False,
        "loginfo": False,
        "time": False,
        "executor": None,
    }
    if parallel is not None:
        if not isinstance(parallel, dict):
            raise TypeError("argument 'parallel' must be of type 'dict'")
        parallel_used.update(parallel)

    if parallel_used.get("time"):
        time_start = time.time()

    if parallel_used.get("executor") is None:
        parallel_used["executor"] = concurrent.futures.ProcessPoolExecutor(
            max_workers=parallel_used.get("max_workers", None)
        )

    with parallel_used.get("executor") as executor:
        fun_jac = EvalParallel(
            fun=fun,
            jac=jac,
            args=args,
            eps=options_used.get("eps"),
            executor=executor,
            forward=parallel_used.get("forward"),
            loginfo=parallel_used.get("loginfo"),
            verbose=parallel_used.get("verbose"),
            n=n,
        )

        if constraints is None:
            out = minimize(
                fun=fun_jac.fun,
                x0=x0,
                jac=fun_jac.jac,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options=options_used,
            )
        else:
            options_used = {
                "disp": None,
                "maxcor": 20,
                "ftol": 1e-03,
                "gtol": 1e-03,
                "eps": 1e-05,
                "maxfun": 100,
                "maxiter": 100,
                "iprint": -1,
                "maxls": 20,
            }

            new_constraints = _convert_jac(x0, constraints, [[-1 for i in range(len(x0))], [1 for i in range(len(x0))]])

            # def callback_function(xk):
            #     cons_loss = 0
            #     for con in new_constraints:
            #         cons_loss += con['fun'](xk)
            #     print('constraints loss: ', cons_loss)
            out = minimize(
                fun=fun_jac.fun,
                x0=x0,
                method="SLSQP",
                jac=fun_jac.jac,
                bounds = Bounds([-1 for i in range(len(x0))], [1 for i in range(len(x0))], keep_feasible=True),
                callback=callback,
                options=options_used,
                constraints=new_constraints
            )

    if parallel_used.get("loginfo"):
        out.loginfo = {
            k: (
                lambda x: np.array(x)
                if isinstance(x[0], list)
                else np.array(x)[np.newaxis].T
            )(v)
            for k, v in fun_jac.info.items()
        }

    if parallel_used.get("time"):
        time_end = time.time()
        out.time = {
            "elapsed": time_end - time_start,
            "step": (time_end - time_start) / out.nfev,
        }

    return out



