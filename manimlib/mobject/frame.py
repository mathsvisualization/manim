from __future__ import annotations

from manimlib.constants import BLACK, GREY_E
from manimlib.constants import FRAME_HEIGHT
from manimlib.mobject.geometry import Rectangle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from manimlib.typing import ManimColor


class ScreenRectangle(Rectangle):
    """
    A rectangle whose width is determined by a specified aspect ratio
    and height.

    ``ScreenRectangle`` extends ``Rectangle`` and is useful for representing
    screens, video frames, or other rectangular regions with a fixed
    width-to-height ratio.

    Parameters
    ----------
    aspect_ratio
        Ratio of the rectangle's width to its height, calculated as
        ``width / height``. Defaults to ``16.0 / 9.0``, corresponding
        to a widescreen 16:9 format.

    height
        Height of the rectangle in scene units. Defaults to ``4``.

    **kwargs
        Additional keyword arguments forwarded to ``Rectangle.__init__``,
        except ``width`` and ``height``, which are explicitly determined
        by ``aspect_ratio`` and ``height`` in this class.

    Notes
    -----
    The rectangle's width is calculated as::

        width = aspect_ratio * height

    Changing the height while keeping the aspect ratio constant scales
    both dimensions proportionally, preserving the specified ratio.

    Examples
    --------
    Create a default widescreen rectangle:

    >>> screen = ScreenRectangle()

    Create a rectangle with a height of 3 scene units:

    >>> screen = ScreenRectangle(height=3)

    Create a rectangle with a 4:3 aspect ratio:

    >>> screen = ScreenRectangle(aspect_ratio=4 / 3, height=3)

    Create a rectangle with customized styling:

    >>> screen = ScreenRectangle(
    ...     aspect_ratio=16 / 9,
    ...     height=4,
    ...     color=BLUE,
    ...     stroke_width=2,
    ... )

    See Also
    --------
    Rectangle
    """

    def __init__(
        self,
        aspect_ratio: float = 16.0 / 9.0,
        height: float = 4,
        **kwargs
    ):
        super().__init__(
            width=aspect_ratio * height,
            height=height,
            **kwargs
        )


class FullScreenRectangle(ScreenRectangle):
    def __init__(
        self,
        height: float = FRAME_HEIGHT,
        fill_color: ManimColor = GREY_E,
        fill_opacity: float = 1,
        stroke_width: float = 0,
        **kwargs,
    ):
        super().__init__(
            height=height,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs
        )


class FullScreenFadeRectangle(FullScreenRectangle):
    def __init__(
        self,
        stroke_width: float = 0.0,
        fill_color: ManimColor = BLACK,
        fill_opacity: float = 0.7,
        **kwargs,
    ):
        super().__init__(
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
        )
