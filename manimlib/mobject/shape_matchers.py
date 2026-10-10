from __future__ import annotations

from colour import Color

from manimlib.config import manim_config
from manimlib.constants import BLACK, RED, YELLOW, DEFAULT_MOBJECT_COLOR
from manimlib.constants import DL, DOWN, DR, LEFT, RIGHT, UL, UR
from manimlib.constants import SMALL_BUFF
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence
    from manimlib.mobject.mobject import Mobject
    from manimlib.typing import ManimColor, Self


class SurroundingRectangle(Rectangle):
    """
    A rectangle that automatically resizes and repositions itself to surround
    a specified mobject, with a configurable buffer around its boundary.

    SurroundingRectangle is useful for highlighting, emphasizing, or visually
    grouping an object in a scene. It inherits the geometry and styling
    capabilities of :class:`Rectangle`.

    Parameters
    ----------
    mobject : Mobject
        The mobject to surround. The rectangle is sized and positioned around
        this object during initialization.
    buff : float, optional
        Extra spacing between the surrounded mobject and the rectangle's
        boundary. Defaults to ``SMALL_BUFF``.
    color : ManimColor, optional
        The rectangle's color. Defaults to ``YELLOW``.
    **kwargs
        Additional keyword arguments forwarded to :class:`Rectangle`, allowing
        other supported rectangle properties to be configured.

    Attributes
    ----------
    mobject : Mobject
        The object currently being surrounded.
    buff : float
        The buffer used when positioning and sizing the rectangle.

    Methods
    -------
    surround(mobject, buff=None)
        Updates the target object and adjusts the rectangle to surround it.
        If ``buff`` is ``None``, the existing buffer is retained. Returns self.

    set_buff(buff)
        Updates the buffer and repositions and resizes the rectangle around
        the currently stored mobject. Returns self.

    Notes
    -----
    - The constructor immediately calls ``surround(mobject)`` to fit the
      rectangle around the target.
    - If the target is fixed in the frame, the rectangle is also fixed in
      the frame using ``fix_in_frame()``.
    - Calling ``surround()`` changes the stored target and can optionally
      change the buffer.
    - Calling ``set_buff()`` changes the spacing without changing the target.
    - The class inherits other geometry and styling methods from
      :class:`Rectangle`.
    - SurroundingRectangle does not automatically track later transformations
      of the target. To update the rectangle, call ``surround()`` again or
      use an appropriate updater.

    Examples
    --------
    Surround a text object with a yellow rectangle::

        text = Tex("Hello, Manim")
        rect = SurroundingRectangle(text)
        self.add(text, rect)

    Choose a custom buffer and color::

        text = Tex("Important")
        rect = SurroundingRectangle(
            text,
            buff=0.2,
            color=RED,
            stroke_width=3,
        )
        self.add(text, rect)

    Change the buffer after construction::

        text = Tex("Math")
        rect = SurroundingRectangle(text, buff=0.1)
        rect.set_buff(0.4)

    Surround a different object::

        first = Circle()
        second = Square()
        rect = SurroundingRectangle(first)
        rect.surround(second, buff=0.2)

    See Also
    --------
    Rectangle
    SurroundingRectangle.surround
    SurroundingRectangle.set_buff
    """

    def __init__(
        self,
        mobject: Mobject,
        buff: float = SMALL_BUFF,
        color: ManimColor = YELLOW,
        **kwargs
    ):
        super().__init__(color=color, **kwargs)
        self.buff = buff
        self.surround(mobject)
        if mobject.is_fixed_in_frame():
            self.fix_in_frame()

    def surround(self, mobject, buff=None) -> Self:
        self.mobject = mobject
        self.buff = buff if buff is not None else self.buff
        super().surround(mobject, self.buff)
        return self

    def set_buff(self, buff) -> Self:
        self.buff = buff
        self.surround(self.mobject)
        return self


class BackgroundRectangle(SurroundingRectangle):
    def __init__(
        self,
        mobject: Mobject,
        color: ManimColor = None,
        stroke_width: float = 0,
        stroke_opacity: float = 0,
        fill_opacity: float = 0.75,
        buff: float = 0,
        **kwargs
    ):
        if color is None:
            color = manim_config.camera.background_color
        super().__init__(
            mobject,
            color=color,
            stroke_width=stroke_width,
            stroke_opacity=stroke_opacity,
            fill_opacity=fill_opacity,
            buff=buff,
            **kwargs
        )
        self.original_fill_opacity = fill_opacity

    def pointwise_become_partial(self, mobject: Mobject, a: float, b: float) -> Self:
        self.set_fill(opacity=b * self.original_fill_opacity)
        return self

    def set_style(
        self,
        stroke_color: ManimColor | None = None,
        stroke_width: float | None = None,
        fill_color: ManimColor | None = None,
        fill_opacity: float | None = None,
        family: bool = True,
        **kwargs
    ) -> Self:
        # Unchangeable style, except for fill_opacity
        VMobject.set_style(
            self,
            stroke_color=BLACK,
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=fill_opacity
        )
        return self

    def get_fill_color(self) -> Color:
        return Color(self.color)


class Cross(VGroup):
    def __init__(
        self,
        mobject: Mobject,
        stroke_color: ManimColor = RED,
        stroke_width: float | Sequence[float] = [0, 6, 0],
        **kwargs
    ):
        super().__init__(
            Line(UL, DR),
            Line(UR, DL),
        )
        self.insert_n_curves(20)
        self.replace(mobject, stretch=True)
        self.set_stroke(stroke_color, width=stroke_width)


class Underline(Line):
    def __init__(
        self,
        mobject: Mobject,
        buff: float = SMALL_BUFF,
        stroke_color=DEFAULT_MOBJECT_COLOR,
        stroke_width: float | Sequence[float] = [0, 3, 3, 0],
        stretch_factor=1.2,
        **kwargs
    ):
        super().__init__(LEFT, RIGHT, **kwargs)
        if not isinstance(stroke_width, (float, int)):
            self.insert_n_curves(len(stroke_width) - 2)
        self.set_stroke(stroke_color, stroke_width)
        self.set_width(mobject.get_width() * stretch_factor)
        self.next_to(mobject, DOWN, buff=buff)
