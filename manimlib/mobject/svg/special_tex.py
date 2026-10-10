from __future__ import annotations

from manimlib.constants import MED_SMALL_BUFF, DEFAULT_MOBJECT_COLOR, GREY_C
from manimlib.constants import DOWN, LEFT, RIGHT, UP
from manimlib.constants import FRAME_WIDTH
from manimlib.constants import MED_LARGE_BUFF, SMALL_BUFF
from manimlib.mobject.geometry import Line
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.svg.tex_mobject import TexText


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manimlib.typing import ManimColor, Vect3


class BulletedList(VGroup):
    """
    Create a vertically arranged list of LaTeX-rendered items.

    `BulletedList` is a subclass of :class:`VGroup` that renders a collection of
    text items using a LaTeX list environment. By default, it creates an unordered
    list using the `itemize` environment. When `numbered=True`, it uses the
    `enumerate` environment instead.

    Each item is prefixed with a LaTeX ``\\item`` command, combined into a single
    LaTeX expression, and rendered through `TexText`. The individual items are
    then extracted as separate mobjects and arranged vertically, allowing them
    to be animated, repositioned, or styled independently.

    Parameters
    ----------
    *items : str
        One or more strings representing the list entries. Each string is inserted
        into the LaTeX list as a separate item. The strings must be valid within
        the generated LaTeX environment.

    buff : float, default=MED_LARGE_BUFF
        Vertical spacing between consecutive list items. Passed to `arrange()`.

    aligned_edge : Vect3, default=LEFT
        Edge used to align the list items during vertical arrangement. By default,
        the left edges of the items are aligned.

    numbered : bool, default=False
        Determines which LaTeX list environment is used:

        - ``False``: use ``itemize`` for an unordered list.
        - ``True``: use ``enumerate`` for a numbered list.

    **kwargs
        Additional keyword arguments forwarded to `TexText`. These can configure
        text rendering, font size, colors, isolation, and other supported options.

    Attributes
    ----------
    submobjects
        The individual rendered list items, stored as children of the `VGroup`.
        Each item is a separately selectable and animatable mobject.

    Methods
    -------
    __init__(*items, buff=MED_LARGE_BUFF, aligned_edge=LEFT,
             numbered=False, **kwargs)
        Construct the LaTeX list, render it, extract each item, and arrange the
        resulting mobjects vertically.

        The constructor performs these steps:

        1. Prefix each supplied item with ``\\item``.
        2. Select the `itemize` or `enumerate` environment according to `numbered`.
        3. Join the generated list entries into one LaTeX expression.
        4. Render the expression using `TexText`, isolating each complete item.
        5. Extract each item as a separate part.
        6. Initialize the parent `VGroup` with those parts.
        7. Arrange the items vertically using the supplied spacing and alignment.

    fade_all_but(index, opacity=0.25, scale_factor=0.7)
        Emphasize one list item while reducing the visibility and size of all
        other items.

        Parameters
        ----------
        index : int
            Zero-based index of the item to emphasize.

        opacity : float, default=0.25
            Fill opacity applied to every non-selected item. The selected item
            retains full fill opacity.

        scale_factor : float, default=0.7
            Relative height factor used for non-selected items compared with the
            tallest item in the list. The selected item uses a factor of `1.0`.

        Behavior
        --------
        The method first finds the maximum height among the first submobjects of
        the list items. It then processes each item:

        - The selected item is assigned full fill opacity and a target height equal
          to the maximum item height.
        - Every other item receives the specified `opacity` and a target height
          equal to `scale_factor` multiplied by the maximum item height.
        - Each item is scaled about its left edge so its first submobject reaches
          the calculated target height.

        This method modifies the existing list items in place and returns `None`.

    Examples
    --------
    Create an unordered list:

    >>> from manimlib import *
    >>> items = BulletedList(
    ...     "Define the problem",
    ...     "Find a solution",
    ...     "Verify the result",
    ... )
    >>> self.add(items)

    Create a numbered list:

    >>> steps = BulletedList(
    ...     "Choose a variable",
    ...     "Write an equation",
    ...     "Solve the equation",
    ...     numbered=True,
    ... )
    >>> self.add(steps)

    Customize spacing and alignment:

    >>> items = BulletedList(
    ...     "First item",
    ...     "Second item",
    ...     "Third item",
    ...     buff=0.5,
    ...     aligned_edge=LEFT,
    ...     font_size=36,
    ... )
    >>> self.add(items)

    Emphasize one item while de-emphasizing the others:

    >>> items = BulletedList(
    ...     "Introduction",
    ...     "Main idea",
    ...     "Conclusion",
    ... )
    >>> items.fade_all_but(1, opacity=0.2, scale_factor=0.7)

    Notes
    -----
    - The `index` argument of `fade_all_but()` is zero-based. For example,
      `index=0` emphasizes the first item.
    - `fade_all_but()` changes fill opacity and scale; it does not explicitly
      change stroke opacity or color.
    - Scaling is performed about each item's left edge, helping preserve the
      alignment of the list while item sizes change.
    - The method assumes that the list is nonempty and that each item has a
      first submobject with a measurable height.
    - Since the list is rendered through LaTeX, the entries must be compatible
      with the selected LaTeX environment and compilation configuration.
    - The list markers are generated by LaTeX rather than by manually created
      bullet or number mobjects.
    """

    def __init__(
        self,
        *items: str,
        buff: float = MED_LARGE_BUFF,
        aligned_edge: Vect3 = LEFT,
        numbered: bool = False,
        **kwargs
    ):
        labelled_content = [R"\item " + item for item in items]
        enum_str = "enumerate" if numbered else "itemize"
        tex_string = "\n".join([
            fR"\begin{{{enum_str}}}",
            *labelled_content,
            fR"\end{{{enum_str}}}"
        ])
        tex_text = TexText(tex_string, isolate=labelled_content, **kwargs)
        lines = (tex_text.select_part(part) for part in labelled_content)

        super().__init__(*lines)

        self.arrange(DOWN, buff=buff, aligned_edge=aligned_edge)

    def fade_all_but(self, index: int, opacity: float = 0.25, scale_factor=0.7) -> None:
        max_dot_height = max([item[0].get_height() for item in self.submobjects])
        for i, part in enumerate(self.submobjects):
            trg_dot_height = (1.0 if i == index else scale_factor) * max_dot_height
            part.set_fill(opacity=(1.0 if i == index else opacity))
            part.scale(trg_dot_height / part[0].get_height(), about_edge=LEFT)


class TexTextFromPresetString(TexText):
    tex: str = ""
    default_color: ManimColor = DEFAULT_MOBJECT_COLOR

    def __init__(self, **kwargs):
        super().__init__(
            self.tex,
            color=kwargs.pop("color", self.default_color),
            **kwargs
        )


class Title(TexText):
    def __init__(
        self,
        *text_parts: str,
        font_size: int = 72,
        include_underline: bool = True,
        underline_width: float = FRAME_WIDTH - 2,
        # This will override underline_width
        match_underline_width_to_text: bool = False,
        underline_buff: float = SMALL_BUFF,
        underline_style: dict = dict(stroke_width=2, stroke_color=GREY_C),
        **kwargs
    ):
        super().__init__(*text_parts, font_size=font_size, **kwargs)
        self.to_edge(UP, buff=MED_SMALL_BUFF)
        if include_underline:
            underline = Line(LEFT, RIGHT, **underline_style)
            underline.next_to(self, DOWN, buff=underline_buff)
            if match_underline_width_to_text:
                underline.match_width(self)
            else:
                underline.set_width(underline_width)
            self.add(underline)
            self.underline = underline
