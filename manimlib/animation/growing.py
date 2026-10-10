from __future__ import annotations

from manimlib.animation.transform import Transform

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from manimlib.mobject.geometry import Arrow
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor


class GrowFromPoint(Transform):
    """
    Animate a mobject growing from a specified point into its original shape.

    GrowFromPoint is a subclass of Transform that creates a starting version
    of the mobject by scaling a copy down to zero size and moving it to a
    specified point. The animation then transforms this collapsed starting
    mobject into the original mobject.

    Optionally, a different color can be applied to the starting mobject.
    This allows the object to grow from a point with one color and transition
    toward its original appearance during the transformation.

    Parameters
    ----------
    mobject : Mobject
        The object to animate. It grows from the specified point into its
        original shape and appearance.

    point : np.ndarray
        The position in space from which the object grows. The starting
        mobject is moved to this point after being scaled to zero.

    point_color : ManimColor, optional
        An optional color applied to the starting mobject. If None, the
        starting mobject retains the color inherited from its copy of the
        original mobject. Defaults to None.

    **kwargs
        Additional keyword arguments forwarded to Transform, such as
        ``run_time`` and other supported transformation settings.

    Methods
    -------
    create_target() -> Mobject
        Return a copy of the original mobject to serve as the transformation
        target.

    create_starting_mobject() -> Mobject
        Create the starting state for the animation by calling the parent
        implementation, scaling the resulting mobject to zero size, moving it
        to ``self.point``, and optionally applying ``self.point_color``.

    Notes
    -----
    - The original mobject is used as the transformation source passed to
      Transform, while ``create_target()`` returns a copy for the target.
    - The starting mobject is scaled to zero before being moved to the
      specified point.
    - ``point_color`` is applied only when it is not None.
    - The animation's interpolation and cleanup behavior are inherited from
      Transform and its parent animation classes.
    - The exact visual result depends on the mobject's geometry, color, and
      the interpolation settings passed through ``kwargs``.

    Examples
    --------
    Example 1: Grow a circle from the origin.

    >>> circle = Circle()
    >>> scene.play(GrowFromPoint(circle, ORIGIN))

    The circle begins collapsed at the origin and transforms into its
    original shape.

    Example 2: Grow a square from a specific position.

    >>> square = Square()
    >>> point = LEFT * 3 + DOWN
    >>> scene.play(GrowFromPoint(square, point))

    The square starts at the specified point and grows into its target shape.
    Here, ``LEFT * 3 + DOWN`` represents a position three units to the left
    and one unit downward from the origin.

    Example 3: Specify the starting color.

    >>> triangle = Triangle(color=BLUE)
    >>> scene.play(
    ...     GrowFromPoint(
    ...         triangle,
    ...         ORIGIN,
    ...         point_color=YELLOW,
    ...     )
    ... )

    The starting mobject is assigned yellow before the transformation begins.
    The target is a copy of the original triangle, so the animation can
    transition from the starting state toward the target's appearance.

    Example 4: Customize the animation duration.

    >>> dot = Dot()
    >>> scene.play(
    ...     GrowFromPoint(
    ...         dot,
    ...         RIGHT * 2,
    ...         run_time=2,
    ...     )
    ... )

    The dot grows from the point two units to the right of the origin over
    two seconds.

    See Also
    --------
    Transform
    GrowFromCenter
    GrowFromEdge
    SpinInFromNothing
    Mobject.scale
    Mobject.move_to
    """

    def __init__(
        self,
        mobject: Mobject,
        point: np.ndarray,
        point_color: ManimColor = None,
        **kwargs
    ):
        self.point = point
        self.point_color = point_color
        super().__init__(mobject, **kwargs)

    def create_target(self) -> Mobject:
        return self.mobject.copy()

    def create_starting_mobject(self) -> Mobject:
        start = super().create_starting_mobject()
        start.scale(0)
        start.move_to(self.point)
        if self.point_color is not None:
            start.set_color(self.point_color)
        return start


class GrowFromCenter(GrowFromPoint):
    def __init__(self, mobject: Mobject, **kwargs):
        point = mobject.get_center()
        super().__init__(mobject, point, **kwargs)


class GrowFromEdge(GrowFromPoint):
    def __init__(self, mobject: Mobject, edge: np.ndarray, **kwargs):
        point = mobject.get_bounding_box_point(edge)
        super().__init__(mobject, point, **kwargs)


class GrowArrow(GrowFromPoint):
    def __init__(self, arrow: Arrow, **kwargs):
        point = arrow.get_start()
        super().__init__(arrow, point, **kwargs)
