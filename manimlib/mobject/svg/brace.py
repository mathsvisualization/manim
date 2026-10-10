from __future__ import annotations

import math
import copy

import numpy as np

from manimlib.constants import DEFAULT_MOBJECT_TO_MOBJECT_BUFF, SMALL_BUFF
from manimlib.constants import DOWN, LEFT, ORIGIN, RIGHT, DL, DR, UL, UP
from manimlib.constants import PI
from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeIn
from manimlib.animation.growing import GrowFromCenter
from manimlib.mobject.svg.tex_mobject import Tex
from manimlib.mobject.svg.tex_mobject import TexText
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.iterables import listify
from manimlib.utils.space_ops import get_norm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable

    from manimlib.animation.animation import Animation
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import Vect3


class Brace(Tex):
    """
    A LaTeX-rendered brace used to annotate a mobject and optionally position
    text or mathematical expressions at the brace's tip.

    Brace inherits from Tex and uses a LaTeX underbrace representation by default.
    It adjusts the brace's width to match the width of a target mobject, positions
    the brace relative to that mobject, and rotates both objects as needed to
    support different brace orientations.

    The class is useful for grouping parts of a mathematical expression, labeling
    a region of a diagram, or indicating which portion of a visual construction
    corresponds to a particular quantity or concept.

    Parameters
    ----------
    mobject : Mobject
        The target mobject whose width determines the initial width of the brace.
        Its lower-left and lower-right corners are used to calculate the target
        width after temporarily rotating it into the brace's reference orientation.

    direction : Vect3, optional
        The direction in which the brace is oriented relative to the target
        mobject. Defaults to DOWN. The direction's first two components determine
        the rotation angle used during positioning.

    buff : float, optional
        The vertical offset applied while positioning the brace relative to the
        target mobject in the brace's temporary reference orientation. Defaults
        to 0.2.

    tex_string : str, optional
        The LaTeX string used to render the brace. Defaults to
        R"\underbrace{\qquad}". The implementation assumes a particular point
        arrangement in this LaTeX representation when identifying the brace's tip
        and adjusting its width.

    **kwargs
        Additional keyword arguments forwarded to Tex during initialization.

    Attributes
    ----------
    tip_point_index : int
        Index of the point in the brace's complete point array that represents
        its tip. It is determined by locating the point with the minimum y
        coordinate in the initially rendered brace geometry. This index is
        subsequently used by get_tip() to retrieve the tip's position.

    Width Adjustment
    ----------------
    The constructor calculates the horizontal distance between the target
    mobject's lower-left and lower-right corners after rotating the target into
    a reference orientation. It then calls set_initial_width() to resize the
    brace to that width before positioning and rotating both objects back.

    set_initial_width() handles widening differently from narrowing. When the
    requested width exceeds the brace's current width, it expands the two
    outer rectangular portions while moving the corresponding tips outward.
    When the requested width is smaller, it stretches the entire brace to the
    requested width.

    Brace Orientation and Placement
    -------------------------------
    The constructor computes a rotation angle from the supplied direction's
    first two components. It temporarily rotates the target mobject, determines
    the target width, positions the brace relative to the target, and finally
    rotates both objects back to their intended orientation.

    This approach allows the brace to be placed along different orientations
    without implementing a separate brace geometry for every direction.

    Methods
    -------
    set_initial_width(width)
        Adjusts the brace to the specified width and returns the brace itself.

    put_at_tip(mob, use_next_to=True, **kwargs)
        Positions another mobject relative to the brace's tip. When use_next_to
        is True, it uses Mobject.next_to() with the rounded brace direction.
        Otherwise, it moves the supplied mobject to the tip and shifts it along
        the brace direction by half its width plus a buffer.

    get_text(text, **kwargs)
        Creates a Text mobject and positions it relative to the brace's tip.
        The optional buff argument controls the spacing and defaults to
        SMALL_BUFF. Other keyword arguments are passed to Text.

    get_tex(*tex, **kwargs)
        Creates a Tex mobject and positions it relative to the brace's tip.
        The optional buff argument controls the spacing and defaults to
        SMALL_BUFF. Other keyword arguments are passed to Tex.

    get_tip()
        Returns the point at tip_point_index from the brace's complete point
        array. The implementation relies on the geometry of the chosen LaTeX
        brace representation.

    get_direction()
        Computes a normalized vector from the brace's center to its tip.
        This vector is used to determine the direction in which labels should
        be positioned relative to the brace.

    Examples
    --------
    Create a brace beneath a mathematical expression:

        expression = Tex("a", "+", "b", "+", "c")
        brace = Brace(expression, direction=DOWN)
        label = brace.get_tex("x")
        self.add(expression, brace, label)

    Create a brace with a custom buffer:

        brace = Brace(expression, direction=DOWN, buff=0.3)
        label = brace.get_text("sum", buff=0.15)
        self.add(expression, brace, label)

    Position an existing mobject at the brace's tip:

        brace.put_at_tip(label)

    Notes
    -----
    - The default LaTeX representation is an underbrace. Other tex_string values
      may not have the same point arrangement or a reliably identifiable tip.
    - tip_point_index is computed from the minimum y coordinate of the initially
      rendered geometry. Its correctness depends on the chosen LaTeX representation.
    - The width adjustment logic assumes a particular internal organization of
      the rendered brace, including specific submobject indices.
    - set_initial_width() stretches existing geometry rather than regenerating
      the LaTeX expression at a new size.
    - get_direction() returns a unit vector from the brace's center toward its
      tip; it is not simply the original direction argument.
    - When use_next_to is False, put_at_tip() calculates its offset using half
      the supplied mobject's width, even when the brace direction is not
      horizontal. This behavior follows the implementation and may affect
      placement for differently oriented braces.
    - The constructor rotates the supplied target mobject during positioning and
      then rotates it back. It therefore modifies the target's geometry as part
      of the placement procedure.

    See Also
    --------
    Tex
        The LaTeX-rendered mobject class from which Brace inherits.
    Text
        The text mobject used by get_text().
    Mobject.next_to
        The relative-positioning method used by put_at_tip().
    """

    def __init__(
        self,
        mobject: Mobject,
        direction: Vect3 = DOWN,
        buff: float = 0.2,
        tex_string: str = R"\underbrace{\qquad}",
        **kwargs
    ):
        super().__init__(tex_string, **kwargs)

        angle = -math.atan2(*direction[:2]) + PI
        mobject.rotate(-angle, about_point=ORIGIN)
        left = mobject.get_corner(DL)
        right = mobject.get_corner(DR)
        target_width = right[0] - left[0]

        self.tip_point_index = np.argmin(self.get_all_points()[:, 1])
        self.set_initial_width(target_width)
        self.shift(left - self.get_corner(UL) + buff * DOWN)
        for mob in mobject, self:
            mob.rotate(angle, about_point=ORIGIN)

    def set_initial_width(self, width: float):
        width_diff = width - self.get_width()
        if width_diff > 0:
            for tip, rect, vect in [(self[0], self[1], RIGHT), (self[5], self[4], LEFT)]:
                rect.set_width(
                    width_diff / 2 + rect.get_width(),
                    about_edge=vect, stretch=True
                )
                tip.shift(-width_diff / 2 * vect)
        else:
            self.set_width(width, stretch=True)
        return self

    def put_at_tip(
        self,
        mob: Mobject,
        use_next_to: bool = True,
        **kwargs
    ):
        if use_next_to:
            mob.next_to(
                self.get_tip(),
                np.round(self.get_direction()),
                **kwargs
            )
        else:
            mob.move_to(self.get_tip())
            buff = kwargs.get("buff", DEFAULT_MOBJECT_TO_MOBJECT_BUFF)
            shift_distance = mob.get_width() / 2.0 + buff
            mob.shift(self.get_direction() * shift_distance)
        return self

    def get_text(self, text: str, **kwargs) -> Text:
        buff = kwargs.pop("buff", SMALL_BUFF)
        text_mob = Text(text, **kwargs)
        self.put_at_tip(text_mob, buff=buff)
        return text_mob

    def get_tex(self, *tex: str, **kwargs) -> Tex:
        buff = kwargs.pop("buff", SMALL_BUFF)
        tex_mob = Tex(*tex, **kwargs)
        self.put_at_tip(tex_mob, buff=buff)
        return tex_mob

    def get_tip(self) -> np.ndarray:
        # Very specific to the LaTeX representation
        # of a brace, but it's the only way I can think
        # of to get the tip regardless of orientation.
        return self.get_all_points()[self.tip_point_index]

    def get_direction(self) -> np.ndarray:
        vect = self.get_tip() - self.get_center()
        return vect / get_norm(vect)


class BraceLabel(VMobject):
    label_constructor: type = Tex

    def __init__(
        self,
        obj: VMobject | list[VMobject],
        text: str | Iterable[str],
        brace_direction: np.ndarray = DOWN,
        label_scale: float = 1.0,
        label_buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFF,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.brace_direction = brace_direction
        self.label_scale = label_scale
        self.label_buff = label_buff

        if isinstance(obj, list):
            obj = VGroup(*obj)
        self.brace = Brace(obj, brace_direction, **kwargs)

        self.label = self.label_constructor(*listify(text), **kwargs)
        self.label.scale(self.label_scale)

        self.brace.put_at_tip(self.label, buff=self.label_buff)
        self.set_submobjects([self.brace, self.label])

    def creation_anim(
        self,
        label_anim: Animation = FadeIn,
        brace_anim: Animation = GrowFromCenter
    ) -> AnimationGroup:
        return AnimationGroup(brace_anim(self.brace), label_anim(self.label))

    def shift_brace(self, obj: VMobject | list[VMobject], **kwargs):
        if isinstance(obj, list):
            obj = VMobject(*obj)
        self.brace = Brace(obj, self.brace_direction, **kwargs)
        self.brace.put_at_tip(self.label)
        self.submobjects[0] = self.brace
        return self

    def change_label(self, *text: str, **kwargs):
        self.label = self.label_constructor(*text, **kwargs)
        if self.label_scale != 1:
            self.label.scale(self.label_scale)

        self.brace.put_at_tip(self.label)
        self.submobjects[1] = self.label
        return self

    def change_brace_label(self, obj: VMobject | list[VMobject], *text: str):
        self.shift_brace(obj)
        self.change_label(*text)
        return self

    def copy(self):
        copy_mobject = copy.copy(self)
        copy_mobject.brace = self.brace.copy()
        copy_mobject.label = self.label.copy()
        copy_mobject.set_submobjects([copy_mobject.brace, copy_mobject.label])

        return copy_mobject


class BraceText(BraceLabel):
    label_constructor: type = TexText


class LineBrace(Brace):
    def __init__(self, line: Line, direction=UP, **kwargs):
        angle = line.get_angle()
        line.rotate(-angle)
        super().__init__(line, direction, **kwargs)
        line.rotate(angle)
        self.rotate(angle, about_point=line.get_center())
