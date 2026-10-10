from __future__ import annotations

from isosurfaces import plot_isoline
import numpy as np

from manimlib.constants import FRAME_X_RADIUS, FRAME_Y_RADIUS
from manimlib.constants import YELLOW
from manimlib.mobject.types.vectorized_mobject import VMobject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Sequence, Tuple
    from manimlib.typing import ManimColor, Vect3


class ParametricCurve(VMobject):
    """
    A vectorized-looking geometric curve constructed by sampling a
    parameterized function over a specified parameter range.

    ``ParametricCurve`` extends ``VMobject`` and generates its points
    by evaluating a function of one parameter, t. The resulting samples
    are connected by straight segments and can optionally be smoothed.
    Known discontinuities can be supplied to split the curve into
    separate paths around those parameter values.

    Parameters
    ----------
    t_func
        Callable that maps a parameter value ``t`` to a sequence of
        coordinates or a three-dimensional vector representing a point
        on the curve.

    t_range
        Tuple ``(t_min, t_max, step)`` specifying the parameter interval
        and sampling step. Defaults to ``(0, 1, 0.1)``.

    epsilon
        Small positive offset used to create boundaries on either side
        of each supplied discontinuity. Defaults to ``1e-8``.

    discontinuities
        Sequence of parameter values at which the curve is discontinuous.
        Values strictly inside the parameter interval are used to split
        the sampled curve. Defaults to an empty sequence.

    use_smoothing
        Whether to smooth the generated paths after sampling. Defaults
        to ``True``.

    **kwargs
        Additional keyword arguments forwarded to ``VMobject.__init__``.

    Attributes
    ----------
    t_func
        Function used to evaluate points on the curve.

    t_range
        Parameter interval and sampling step used during initialization.

    epsilon
        Offset used around discontinuities.

    discontinuities
        Sequence of discontinuity values supplied to the constructor.

    use_smoothing
        Whether smoothing is enabled during point initialization.

    Notes
    -----
    During ``init_points``, discontinuities outside or at the endpoints
    of the parameter interval are ignored. The remaining discontinuities
    are offset by ``epsilon`` to create boundary times. These times are
    sorted and paired to determine the intervals sampled into separate
    paths.

    For each interval, parameter values are generated using
    ``numpy.arange`` and the interval endpoint is appended explicitly.
    The function is evaluated at each sample, and consecutive points
    are connected by straight segments.

    If smoothing is enabled, ``make_smooth(approx=True)`` is called
    after the paths are constructed. If no points were generated, the
    curve falls back to a single point evaluated at the starting
    parameter value.

    ``get_point_from_function`` evaluates ``t_func`` at a single
    parameter and converts its result to a NumPy array.

    ``get_function`` returns ``underlying_function`` if present;
    otherwise, it returns ``function`` if that attribute exists.
    If neither attribute exists, it returns ``None``.

    ``get_x_range`` returns the ``x_range`` attribute if present,
    or ``None`` otherwise. This attribute is not assigned by the
    constructor itself.

    Examples
    --------
    Create a parametric curve representing a circle:

    >>> curve = ParametricCurve(
    ...     lambda t: np.array([
    ...         np.cos(t),
    ...         np.sin(t),
    ...         0,
    ...     ]),
    ...     t_range=(0, TAU, 0.05),
    ... )

    Create a three-dimensional helix:

    >>> curve = ParametricCurve(
    ...     lambda t: np.array([
    ...         np.cos(t),
    ...         np.sin(t),
    ...         t / 2,
    ...     ]),
    ...     t_range=(0, 4 * PI, 0.1),
    ... )

    Disable smoothing:

    >>> curve = ParametricCurve(
    ...     lambda t: np.array([t, t**2, 0]),
    ...     t_range=(-2, 2, 0.1),
    ...     use_smoothing=False,
    ... )

    Specify a known discontinuity:

    >>> curve = ParametricCurve(
    ...     lambda t: np.array([t, 1 / t, 0]),
    ...     t_range=(-1, 1, 0.05),
    ...     discontinuities=[0],
    ... )

    Retrieve the parameter function:

    >>> curve = ParametricCurve(lambda t: np.array([t, t**2, 0]))
    >>> function = curve.get_t_func()

    See Also
    --------
    VMobject
    ParametricSurface
    """

    def __init__(
        self,
        t_func: Callable[[float], Sequence[float] | Vect3],
        t_range: Tuple[float, float, float] = (0, 1, 0.1),
        epsilon: float = 1e-8,
        # TODO, automatically figure out discontinuities
        discontinuities: Sequence[float] = [],
        use_smoothing: bool = True,
        **kwargs
    ):
        self.t_func = t_func
        self.t_range = t_range
        self.epsilon = epsilon
        self.discontinuities = discontinuities
        self.use_smoothing = use_smoothing
        super().__init__(**kwargs)

    def get_point_from_function(self, t: float) -> Vect3:
        return np.array(self.t_func(t))

    def init_points(self):
        t_min, t_max, step = self.t_range

        jumps = np.array(self.discontinuities)
        jumps = jumps[(jumps > t_min) & (jumps < t_max)]
        boundary_times = [t_min, t_max, *(jumps - self.epsilon), *(jumps + self.epsilon)]
        boundary_times.sort()
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            t_range = [*np.arange(t1, t2, step), t2]
            points = np.array([self.t_func(t) for t in t_range])
            self.start_new_path(points[0])
            self.add_points_as_corners(points[1:])
        if self.use_smoothing:
            self.make_smooth(approx=True)
        if not self.has_points():
            self.set_points(np.array([self.t_func(t_min)]))
        return self

    def get_t_func(self):
        return self.t_func

    def get_function(self):
        if hasattr(self, "underlying_function"):
            return self.underlying_function
        if hasattr(self, "function"):
            return self.function

    def get_x_range(self):
        if hasattr(self, "x_range"):
            return self.x_range


class FunctionGraph(ParametricCurve):
    def __init__(
        self,
        function: Callable[[float], float],
        x_range: Tuple[float, float, float] = (-8, 8, 0.25),
        color: ManimColor = YELLOW,
        **kwargs
    ):
        self.function = function
        self.x_range = x_range

        def parametric_function(t):
            return [t, function(t), 0]

        super().__init__(parametric_function, self.x_range, **kwargs)


class ImplicitFunction(VMobject):
    def __init__(
        self,
        func: Callable[[float, float], float],
        x_range: Tuple[float, float] = (-FRAME_X_RADIUS, FRAME_X_RADIUS),
        y_range: Tuple[float, float] = (-FRAME_Y_RADIUS, FRAME_Y_RADIUS),
        min_depth: int = 5,
        max_quads: int = 1500,
        use_smoothing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        p_min, p_max = (
            np.array([x_range[0], y_range[0]]),
            np.array([x_range[1], y_range[1]]),
        )
        curves = plot_isoline(
            fn=lambda u: func(u[0], u[1]),
            pmin=p_min,
            pmax=p_max,
            min_depth=min_depth,
            max_quads=max_quads,
        )  # returns a list of lists of 2D points
        curves = [
            np.pad(curve, [(0, 0), (0, 1)])
            for curve in curves
            if curve != []
        ]  # add z coord as 0
        for curve in curves:
            self.start_new_path(curve[0])
            self.add_points_as_corners(curve[1:])
        if use_smoothing:
            self.make_smooth()
