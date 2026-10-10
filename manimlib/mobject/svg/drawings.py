from __future__ import annotations

import numpy as np
import itertools as it
import random
import math

from manimlib.animation.composition import AnimationGroup
from manimlib.animation.rotation import Rotating
from manimlib.constants import BLACK
from manimlib.constants import BLUE_A
from manimlib.constants import BLUE_B
from manimlib.constants import BLUE_C
from manimlib.constants import BLUE_D
from manimlib.constants import DOWN
from manimlib.constants import DOWN
from manimlib.constants import FRAME_WIDTH
from manimlib.constants import GREEN
from manimlib.constants import GREEN_SCREEN
from manimlib.constants import GREEN_E
from manimlib.constants import GREY
from manimlib.constants import GREY_A
from manimlib.constants import GREY_B
from manimlib.constants import GREY_E
from manimlib.constants import LEFT
from manimlib.constants import LEFT
from manimlib.constants import MED_LARGE_BUFF
from manimlib.constants import MED_SMALL_BUFF
from manimlib.constants import ORIGIN
from manimlib.constants import OUT
from manimlib.constants import PI
from manimlib.constants import RED
from manimlib.constants import RED_E
from manimlib.constants import RIGHT
from manimlib.constants import SMALL_BUFF
from manimlib.constants import SMALL_BUFF
from manimlib.constants import UP
from manimlib.constants import UL
from manimlib.constants import UR
from manimlib.constants import DL
from manimlib.constants import DR
from manimlib.constants import WHITE
from manimlib.constants import YELLOW
from manimlib.constants import TAU
from manimlib.mobject.boolean_ops import Difference
from manimlib.mobject.boolean_ops import Union
from manimlib.mobject.geometry import Arc
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Dot
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Polygon
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import Square
from manimlib.mobject.geometry import AnnularSector
from manimlib.mobject.numbers import Integer
from manimlib.mobject.shape_matchers import SurroundingRectangle
from manimlib.mobject.svg.svg_mobject import SVGMobject
from manimlib.mobject.svg.special_tex import TexTextFromPresetString
from manimlib.mobject.three_dimensions import Prismify
from manimlib.mobject.three_dimensions import VCube
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.mobject.svg.text_mobject import Text
from manimlib.utils.bezier import interpolate
from manimlib.utils.iterables import adjacent_pairs
from manimlib.utils.rate_functions import linear
from manimlib.utils.space_ops import angle_of_vector
from manimlib.utils.space_ops import compass_directions
from manimlib.utils.space_ops import get_norm
from manimlib.utils.space_ops import midpoint
from manimlib.utils.space_ops import rotate_vector

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Sequence, Callable
    from manimlib.typing import ManimColor, Vect3


class Checkmark(TexTextFromPresetString):
    """
    A preset LaTeX text mobject that displays a checkmark symbol.

    Checkmark inherits from TexTextFromPresetString and uses the LaTeX command
    R"\ding{51}" to render a checkmark. Its default color is GREEN, making it
    suitable for indicating correctness, completion, approval, or a successful
    condition in a scene.

    The class provides a convenient alternative to manually creating a text
    mobject containing the checkmark command. Since the symbol and default color
    are defined as class attributes, instances use this preset configuration
    unless the inherited implementation or supplied arguments override it.

    Class Attributes
    ----------------
    tex : str
        LaTeX command R"\ding{51}", used to render the checkmark symbol.

    default_color : ManimColor
        Defaults to GREEN.

    Example
    -------
        check = Checkmark()
        self.add(check)

    Display a checkmark beside a statement:

        statement = TexText("Correct")
        check = Checkmark()
        check.next_to(statement, RIGHT)
        self.add(statement, check)

    Notes
    -----
    - The symbol is rendered through the inherited TexTextFromPresetString
      implementation.
    - Rendering the dingbat symbol depends on the required LaTeX package and
      the availability of the corresponding symbol in the TeX environment.
    - GREEN is the default color; the final appearance may depend on inherited
      color-handling behavior and constructor arguments.
    """

    tex: str = R"\ding{51}"
    default_color: ManimColor = GREEN


class Exmark(TexTextFromPresetString):
    """
    A preset LaTeX text mobject that displays an X-shaped cross symbol.

    Exmark inherits from TexTextFromPresetString and uses the LaTeX command
    R"\ding{55}" to render the symbol. Its default color is RED, making it
    suitable for indicating an incorrect answer, a failed condition, rejection,
    or an unsuccessful result in a scene.

    The class provides a convenient way to display this preset symbol without
    manually constructing a text mobject for the corresponding LaTeX command.

    Class Attributes
    ----------------
    tex : str
        LaTeX command R"\ding{55}", used to render the X-shaped symbol.

    default_color : ManimColor
        Defaults to RED.

    Example
    -------
        mark = Exmark()
        self.add(mark)

    Display the symbol beside an incorrect answer:

        answer = TexText("Incorrect")
        mark = Exmark()
        mark.next_to(answer, RIGHT)
        self.add(answer, mark)

    Notes
    -----
    - The symbol is rendered through the inherited TexTextFromPresetString
      implementation.
    - Rendering the dingbat symbol depends on the required LaTeX package and
      the availability of the corresponding symbol in the TeX environment.
    - RED is the default color; the final appearance may depend on inherited
      color-handling behavior and constructor arguments.
    """

    tex: str = R"\ding{55}"
    default_color: ManimColor = RED


class Lightbulb(SVGMobject):
    """
    An SVG-based lightbulb mobject for visually representing ideas, insights,
    creativity, or a source of illumination in a ManimGL scene.

    Lightbulb inherits from SVGMobject and loads the SVG asset identified by
    the class attribute file_name = "lightbulb". It exposes convenient
    parameters for controlling the symbol's height, color, stroke width, and
    fill opacity.

    After initializing the SVG mobject, the constructor calls insert_n_curves(25)
    to increase the number of curves in its vector geometry. This can provide
    additional curve segments for operations that work on the mobject's paths,
    depending on the underlying SVGMobject implementation.

    Parameters
    ----------
    height : float, optional
        Desired height of the lightbulb mobject. Defaults to 1.0.

    color : ManimColor, optional
        Color applied to the SVG mobject. Defaults to YELLOW.

    stroke_width : float, optional
        Width of the SVG outlines. Defaults to 3.0.

    fill_opacity : float, optional
        Opacity of the SVG fill. Defaults to 0.0, making the fill fully
        transparent under the usual opacity convention.

    **kwargs
        Additional keyword arguments forwarded to SVGMobject.

    Class Attributes
    ----------------
    file_name : str
        Set to "lightbulb", identifying the SVG asset loaded by the parent
        class. The actual file resolution depends on SVGMobject's asset-loading
        implementation.

    Examples
    --------
    Create a default lightbulb:

        bulb = Lightbulb()
        self.add(bulb)

    Change its size and color:

        bulb = Lightbulb(height=2.0, color=YELLOW)
        self.add(bulb)

    Use a transparent fill with a visible outline:

        bulb = Lightbulb(
            height=1.5,
            color=BLUE,
            stroke_width=4.0,
            fill_opacity=0.0,
        )
        self.add(bulb)

    Notes
    -----
    - Lightbulb is an SVG-based vector mobject, not a raster image.
    - The availability and appearance of the graphic depend on the SVG asset
      referenced by file_name.
    - insert_n_curves(25) modifies the vector geometry after initialization.
      Its exact effect depends on how SVGMobject implements curve subdivision.
    - A fill_opacity of 0.0 makes the fill transparent; it does not by itself
      guarantee that every part of the SVG is outline-only, since the SVG's
      paths and inherited styling also affect rendering.
    """

    file_name = "lightbulb"

    def __init__(
        self,
        height: float = 1.0,
        color: ManimColor = YELLOW,
        stroke_width: float = 3.0,
        fill_opacity: float = 0.0,
        **kwargs
    ):
        super().__init__(
            height=height,
            color=color,
            stroke_width=stroke_width,
            fill_opacity=fill_opacity,
            **kwargs
        )
        self.insert_n_curves(25)


class Speedometer(VMobject):
    """
    A visual speedometer mobject consisting of an arc, tick marks, numeric labels,
    and a rotatable needle.

    Speedometer inherits from VMobject and constructs a gauge-like display for
    representing a numeric velocity value. The gauge contains an arc spanning
    the configured angular range, evenly spaced tick marks, labels increasing
    in increments of ten, and a triangular needle that indicates the current
    reading.

    The needle can be rotated directly or positioned to represent a velocity
    using move_needle_to_velocity(). The latter maps the supplied velocity to
    an angular position based on the number of ticks and the total arc angle.

    Parameters
    ----------
    arc_angle : float, optional
        Total angular span of the gauge arc, in radians. Defaults to 4 * PI / 3,
        corresponding to 240 degrees. The arc is centered around the upward
        direction.

    num_ticks : int, optional
        Number of tick marks and numeric labels distributed across the arc.
        Defaults to 8. The labels are generated as 0, 10, 20, and so on.

    tick_length : float, optional
        Length of each tick mark, measured radially in the gauge's local
        coordinate system. Defaults to 0.2. It also determines the height of
        each numeric label and contributes to the labels' placement.

    needle_width : float, optional
        Width of the triangular needle after it is stretched to fit the
        requested width. Defaults to 0.1.

    needle_height : float, optional
        Height of the triangular needle after it is stretched to fit the
        requested height. Defaults to 0.8.

    needle_color : ManimColor, optional
        Fill color of the needle. Defaults to YELLOW.

    **kwargs
        Additional keyword arguments forwarded to VMobject initialization.

    Attributes
    ----------
    arc_angle : float
        Angular span of the gauge arc, in radians.

    num_ticks : int
        Number of tick marks and numeric labels.

    tick_length : float
        Radial length of the tick marks.

    needle_width : float
        Configured width of the needle.

    needle_height : float
        Configured height of the needle.

    needle_color : ManimColor
        Configured fill color of the needle.

    arc : Arc
        Arc forming the curved scale of the speedometer.

    needle : Polygon
        Triangular polygon that indicates the current reading.

    center_offset : np.ndarray
        Center position recorded after the gauge has been constructed. The
        overridden get_center() method uses this offset to provide a stable
        reference center relative to the initial geometry.

    Construction
    ------------
    The arc begins at PI / 2 + arc_angle / 2 and extends through a negative
    angle of arc_angle, ending at PI / 2 - arc_angle / 2. This places the
    gauge's angular range symmetrically around the upward direction.

    The tick marks are positioned using evenly spaced angles generated by
    numpy.linspace(). Each tick extends radially from (1 - tick_length) times
    its direction vector to the direction vector itself.

    Each tick receives an Integer label equal to ten times its zero-based
    index. The label is resized to tick_length in height and placed farther
    from the origin than the tick mark.

    The needle is created from a triangular Polygon, stretched to the requested
    width and height, and rotated to align with the starting angle of the gauge.
    The resulting polygon is stored in self.needle for subsequent manipulation.

    Methods
    -------
    get_center()
        Returns the center of the VMobject's current geometry, adjusted by
        center_offset when that attribute exists. This compensates for the
        initial center recorded during construction and provides a reference
        center for needle-angle calculations.

    get_needle_tip()
        Returns the second anchor point from the needle's anchor array. Given
        the needle polygon's construction, this point represents its intended
        tip. The result is a point in the scene's coordinate system.

    get_needle_angle()
        Calculates the angle, in radians, of the vector from the speedometer's
        adjusted center to the needle tip. The angle is obtained using
        angle_of_vector().

    rotate_needle(angle)
        Rotates the needle by the specified angle in radians around the center
        of the arc, then returns self. The argument is a relative rotation,
        not an absolute target angle.

    move_needle_to_velocity(velocity)
        Maps a velocity value to a target needle angle and rotates the needle
        to that position. The maximum velocity is calculated as
        10 * (num_ticks - 1). The input is divided by this maximum to obtain
        a proportion of the gauge's range. The target angle is then calculated
        by moving from the starting angle through arc_angle times that
        proportion. The method returns self.

    Examples
    --------
    Create a speedometer with the default configuration:

        speedometer = Speedometer()
        self.add(speedometer)

    Move the needle to represent a velocity:

        speedometer.move_needle_to_velocity(40)

    Rotate the needle by a relative angle:

        speedometer.rotate_needle(PI / 6)

    Create a gauge with a different angular range and needle color:

        speedometer = Speedometer(
            arc_angle=PI,
            num_ticks=6,
            needle_color=RED,
        )
        self.add(speedometer)

    Notes
    -----
    - All angular values are expressed in radians.
    - The default configuration has eight ticks, with labels from 0 to 70.
    - The maximum velocity used by move_needle_to_velocity() is derived from
      num_ticks. It is not an independently configurable parameter.
    - The velocity-to-angle mapping is linear. The method does not clamp
      velocity values to the nominal range, so values outside the range can
      rotate the needle beyond the endpoints of the arc.
    - A zero or single tick count makes the velocity mapping invalid because
      max_velocity becomes zero. A practical configuration should use at least
      two ticks.
    - Numeric labels are generated from tick indices rather than from a
      separately supplied scale.
    - The adjusted center is based on center_offset captured after construction.
      Subsequent transformations of the entire mobject may affect how this
      reference behaves, depending on the base VMobject transformation logic.
    - rotate_needle() rotates around arc.get_arc_center(), whereas
      get_needle_angle() measures the needle angle relative to get_center().
      These centers are expected to coincide for the intended gauge layout.

    See Also
    --------
    Arc
        Creates the curved scale of the gauge.
    Line
        Represents the radial tick marks.
    Integer
        Displays the numeric tick labels.
    Polygon
        Defines the triangular needle.
    """

    def __init__(
        self,
        arc_angle: float = 4 * PI / 3,
        num_ticks: int = 8,
        tick_length: float = 0.2,
        needle_width: float = 0.1,
        needle_height: float = 0.8,
        needle_color: ManimColor = YELLOW,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.arc_angle = arc_angle
        self.num_ticks = num_ticks
        self.tick_length = tick_length
        self.needle_width = needle_width
        self.needle_height = needle_height
        self.needle_color = needle_color

        start_angle = PI / 2 + arc_angle / 2
        end_angle = PI / 2 - arc_angle / 2
        self.arc = Arc(
            start_angle=start_angle,
            angle=-self.arc_angle
        )
        self.add(self.arc)
        tick_angle_range = np.linspace(start_angle, end_angle, num_ticks)
        for index, angle in enumerate(tick_angle_range):
            vect = rotate_vector(RIGHT, angle)
            tick = Line((1 - tick_length) * vect, vect)
            label = Integer(10 * index)
            label.set_height(tick_length)
            label.shift((1 + tick_length) * vect)
            self.add(tick, label)

        needle = Polygon(
            LEFT, UP, RIGHT,
            stroke_width=0,
            fill_opacity=1,
            fill_color=self.needle_color
        )
        needle.stretch_to_fit_width(needle_width)
        needle.stretch_to_fit_height(needle_height)
        needle.rotate(start_angle - np.pi / 2, about_point=ORIGIN)
        self.add(needle)
        self.needle = needle

        self.center_offset = self.get_center()

    def get_center(self):
        result = VMobject.get_center(self)
        if hasattr(self, "center_offset"):
            result -= self.center_offset
        return result

    def get_needle_tip(self):
        return self.needle.get_anchors()[1]

    def get_needle_angle(self):
        return angle_of_vector(
            self.get_needle_tip() - self.get_center()
        )

    def rotate_needle(self, angle):
        self.needle.rotate(angle, about_point=self.arc.get_arc_center())
        return self

    def move_needle_to_velocity(self, velocity):
        max_velocity = 10 * (self.num_ticks - 1)
        proportion = float(velocity) / max_velocity
        start_angle = np.pi / 2 + self.arc_angle / 2
        target_angle = start_angle - self.arc_angle * proportion
        self.rotate_needle(target_angle - self.get_needle_angle())
        return self


class Laptop(VGroup):
    """
    A three-dimensional laptop model composed of a base, keyboard, hinged screen,
    and hinge-axis indicator.

    Laptop inherits from VGroup and constructs a stylized laptop using ManimGL
    vector mobjects. The base is created from a VCube and reshaped according to
    the requested body dimensions. A grid of square mobjects represents the
    keyboard, while a thin screen plate supports a black rectangular screen.

    The screen plate is positioned at the back of the laptop body and rotated
    around its bottom edge to represent an open laptop. The open angle can be
    customized to create different screen positions.

    The assembled laptop is stored as a group, allowing it to be positioned,
    scaled, rotated, and animated as a single object. References to the screen,
    screen plate, and hinge-axis indicator are also exposed for individual
    manipulation.

    Parameters
    ----------
    width : float, optional
        Target width of the laptop body after its initial dimensions have been
        applied. Defaults to 3. The body is resized to this width while
        preserving the proportions established by its preceding stretch
        operations.

    body_dimensions : Tuple[float, float, float], optional
        Initial scale dimensions applied to the VCube along its three coordinate
        axes, in x-, y-, and z-dimension order. Defaults to (4.0, 3.0, 0.05).
        These values determine the base's proportions before its width is set
        to the requested width.

    screen_thickness : float, optional
        Thickness used when reshaping the screen plate along the z-axis.
        Defaults to 0.01. The implementation calculates a stretch factor by
        dividing this value by body_dimensions[2].

    keyboard_width_to_body_width : float, optional
        Ratio between the keyboard's target width and the laptop body's width.
        Defaults to 0.9.

    keyboard_height_to_body_height : float, optional
        Ratio between the keyboard's target height and the laptop body's height.
        Defaults to 0.5.

    screen_width_to_screen_plate_width : float, optional
        Scale factor applied to the screen rectangle relative to the screen
        plate's width. Defaults to 0.9.

    key_color_kwargs : dict, optional
        Keyword arguments passed to each Square used to construct the keyboard
        keys. Defaults to a dictionary with stroke_width=0, fill_color=BLACK,
        and fill_opacity=1. These settings produce solid black keys without
        visible outlines under the usual rendering conventions.

    fill_opacity : float, optional
        Accepted as a constructor parameter, with a default of 1.0. In this
        implementation, the value is not explicitly applied to the body, screen,
        keyboard, or other generated components.

    stroke_width : float, optional
        Accepted as a constructor parameter, with a default of 0.0. In this
        implementation, the value is not explicitly applied to the generated
        components as a general stroke-width setting.

    body_color : ManimColor, optional
        Fill color applied to the final submobject in the sorted body geometry.
        Defaults to GREY_B.

    shaded_body_color : ManimColor, optional
        Fill color initially applied to the body geometry. Defaults to GREY.
        The body is sorted by the z-coordinate of its points before the final
        submobject receives body_color.

    open_angle : float, optional
        Angle, in radians, through which the screen plate is rotated around the
        RIGHT axis at its bottom edge. Defaults to pi / 4, or 45 degrees.

    **kwargs
        Additional keyword arguments forwarded to VGroup initialization.

    Attributes
    ----------
    screen_plate : VGroup or VMobject
        The copied body geometry used as the laptop's screen frame or plate.
        The screen rectangle is added to this object before the plate is rotated.

    screen : Rectangle
        Black rectangular mobject representing the laptop display. It is exposed
        separately so it can be recolored, replaced, or otherwise manipulated.

    axis : Line
        A black line drawn between the body's upper-left-front and
        upper-right-front corners. It serves as a visual indicator of the
        laptop's hinge axis.

    Construction
    ------------
    1. Create a unit VCube and stretch it along each coordinate dimension using
       body_dimensions.
    2. Set the body's width to width, then apply the shaded body fill color.
    3. Sort the body's submobjects according to their point z-coordinates and
       apply body_color to the final submobject.
    4. Copy the body to create the screen plate.
    5. Construct a keyboard from four rows of Square mobjects. Alternate row
       lengths between 12 and 11 keys, arrange the keys horizontally, and arrange
       the rows vertically.
    6. Resize the keyboard using the width and height ratios, position it just
       above the body, and add it to the body group.
    7. Adjust the screen plate's thickness, create a black Rectangle, and fit
       the rectangle to the plate before applying the screen width ratio.
    8. Position the screen in front of the plate, add it to the plate, and place
       the plate at the back edge of the body.
    9. Rotate the plate around its bottom edge by open_angle to represent the
       open display.
    10. Create the hinge-axis Line and add the body, screen plate, and axis to
        the Laptop group.

    Examples
    --------
    Create a laptop using the default dimensions:

        laptop = Laptop()
        self.add(laptop)

    Create a wider laptop:

        laptop = Laptop(width=4.0)
        self.add(laptop)

    Customize the screen opening angle:

        laptop = Laptop(open_angle=PI / 3)
        self.add(laptop)

    Customize the body and screen colors:

        laptop = Laptop(
            body_color=BLUE,
            shaded_body_color=GREY,
            open_angle=PI / 4,
        )
        self.add(laptop)

    Access and manipulate the screen separately:

        laptop = Laptop()
        laptop.screen.set_fill(WHITE, opacity=1)
        self.add(laptop)

    Notes
    -----
    - The laptop is a stylized vector construction, not a detailed physical
      model. Its appearance depends on the dimensions, colors, and rendering
      behavior of the component mobjects.
    - body_dimensions[2] must be nonzero because it is used as the denominator
      when calculating the screen plate's thickness stretch factor.
    - body_dimensions is expected to contain three values corresponding to the
      three coordinate dimensions.
    - The keyboard contains four rows with alternating counts of 12 and 11
      square keys. It does not model a complete physical keyboard layout.
    - The screen is a black Rectangle fitted to the screen plate before being
      scaled to screen_width_to_screen_plate_width.
    - The screen plate rotates around its bottom edge. The resulting orientation
      depends on the initial geometry and the supplied open_angle.
    - fill_opacity and stroke_width are accepted but are not explicitly applied
      to the constructed components by this implementation.
    - body_color is applied to body[-1] after sorting; the exact face or component
      receiving that color depends on the submobject structure of VCube and the
      behavior of sort().
    - The hinge-axis indicator is added as a separate group member and is not
      itself rotated as part of the screen-plate rotation.
    - The body, screen plate, and axis are all included in the parent VGroup, so
      transformations applied to the complete Laptop affect these components
      together.

    See Also
    --------
    VGroup
        Groups multiple mobjects into one composite object.
    VCube
        Supplies the three-dimensional geometry for the laptop body.
    Rectangle
        Represents the laptop screen.
    Square
        Supplies the keyboard key geometry.
    Line
        Represents the hinge-axis indicator.
    """

    def __init__(
        self,
        width: float = 3,
        body_dimensions: Tuple[float, float, float] = (4.0, 3.0, 0.05),
        screen_thickness: float = 0.01,
        keyboard_width_to_body_width: float = 0.9,
        keyboard_height_to_body_height: float = 0.5,
        screen_width_to_screen_plate_width: float = 0.9,
        key_color_kwargs: dict = dict(
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1,
        ),
        fill_opacity: float = 1.0,
        stroke_width: float = 0.0,
        body_color: ManimColor = GREY_B,
        shaded_body_color: ManimColor = GREY,
        open_angle: float = np.pi / 4,
        **kwargs
    ):
        super().__init__(**kwargs)

        body = VCube(side_length=1)
        for dim, scale_factor in enumerate(body_dimensions):
            body.stretch(scale_factor, dim=dim)
        body.set_width(width)
        body.set_fill(shaded_body_color, opacity=1)
        body.sort(lambda p: p[2])
        body[-1].set_fill(body_color)
        screen_plate = body.copy()
        keyboard = VGroup(*[
            VGroup(*[
                Square(**key_color_kwargs)
                for x in range(12 - y % 2)
            ]).arrange(RIGHT, buff=SMALL_BUFF)
            for y in range(4)
        ]).arrange(DOWN, buff=MED_SMALL_BUFF)
        keyboard.stretch_to_fit_width(
            keyboard_width_to_body_width * body.get_width(),
        )
        keyboard.stretch_to_fit_height(
            keyboard_height_to_body_height * body.get_height(),
        )
        keyboard.next_to(body, OUT, buff=0.1 * SMALL_BUFF)
        keyboard.shift(MED_SMALL_BUFF * UP)
        body.add(keyboard)

        screen_plate.stretch(screen_thickness /
                             body_dimensions[2], dim=2)
        screen = Rectangle(
            stroke_width=0,
            fill_color=BLACK,
            fill_opacity=1,
        )
        screen.replace(screen_plate, stretch=True)
        screen.scale(screen_width_to_screen_plate_width)
        screen.next_to(screen_plate, OUT, buff=0.1 * SMALL_BUFF)
        screen_plate.add(screen)
        screen_plate.next_to(body, UP, buff=0)
        screen_plate.rotate(
            open_angle, RIGHT,
            about_point=screen_plate.get_bottom()
        )
        self.screen_plate = screen_plate
        self.screen = screen

        axis = Line(
            body.get_corner(UP + LEFT + OUT),
            body.get_corner(UP + RIGHT + OUT),
            color=BLACK,
            stroke_width=2
        )
        self.axis = axis

        self.add(body, screen_plate, axis)


class VideoIcon(SVGMobject):
    """
    An SVG-based icon representing video content.

    VideoIcon inherits from SVGMobject and loads the SVG asset identified by
    file_name = "video_icon". It provides a simple way to display a video-related
    icon in a ManimGL scene, with configurable width and color.

    Parameters
    ----------
    width : float, optional
        Desired width of the icon. Defaults to 1.2.

    color : optional
        Color applied to the SVG mobject. Defaults to BLUE_A.

    **kwargs
        Additional keyword arguments forwarded to SVGMobject.

    Examples
    --------
        icon = VideoIcon()
        self.add(icon)

        icon = VideoIcon(width=2.0, color=RED)
        self.add(icon)

    Notes
    -----
    - The icon's appearance depends on the SVG asset and SVGMobject's rendering.
    - set_width() adjusts the icon to the requested width while normally
      preserving its aspect ratio.
    """

    file_name: str = "video_icon"

    def __init__(
        self,
        width: float = 1.2,
        color=BLUE_A,
        **kwargs
    ):
        super().__init__(color=color, **kwargs)
        self.set_width(width)


class VideoSeries(VGroup):
    def __init__(
        self,
        num_videos: int = 11,
        gradient_colors: Sequence[ManimColor] = [BLUE_B, BLUE_D],
        width: float = FRAME_WIDTH - MED_LARGE_BUFF,
        **kwargs
    ):
        super().__init__(
            *(VideoIcon() for x in range(num_videos)),
            **kwargs
        )
        self.arrange(RIGHT)
        self.set_width(width)
        self.set_color_by_gradient(*gradient_colors)


class Clock(VGroup):
    def __init__(
        self,
        stroke_color: ManimColor = WHITE,
        stroke_width: float = 3.0,
        hour_hand_height: float = 0.3,
        minute_hand_height: float = 0.6,
        tick_length: float = 0.1,
        **kwargs,
    ):
        style = dict(stroke_color=stroke_color, stroke_width=stroke_width)
        circle = Circle(**style)
        self.ticks = VGroup()
        for x, angle in enumerate(np.arange(0, TAU, TAU / 12)):
            point = math.cos(angle) * UP + math.sin(angle) * RIGHT
            length = tick_length
            if x % 3 == 0:
                length *= 2
            self.ticks.add(Line(point, (1 - length) * point, **style))
        self.hour_hand = Line(ORIGIN, hour_hand_height * UP, **style)
        self.minute_hand = Line(ORIGIN, minute_hand_height * UP, **style)

        super().__init__(
            circle, self.hour_hand, self.minute_hand, self.ticks
        )


class ClockPassesTime(AnimationGroup):
    def __init__(
        self,
        clock: Clock,
        run_time: float = 5.0,
        hours_passed: float = 12.0,
        rate_func: Callable[[float], float] = linear,
        **kwargs
    ):
        rot_kwargs = dict(
            axis=OUT,
            about_point=clock.get_center()
        )
        hour_radians = -hours_passed * 2 * PI / 12
        super().__init__(
            Rotating(
                clock.hour_hand,
                angle=hour_radians,
                **rot_kwargs
            ),
            Rotating(
                clock.minute_hand,
                angle=12 * hour_radians,
                **rot_kwargs
            ),
            group=clock,
            run_time=run_time,
            **kwargs
        )


class Bubble(VGroup):
    file_name: str = "Bubbles_speech.svg"
    bubble_center_adjustment_factor = 0.125

    def __init__(
        self,
        content: str | VMobject | None = None,
        buff: float = 1.0,
        filler_shape: Tuple[float, float] = (3.0, 2.0),
        pin_point: Vect3 | None = None,
        direction: Vect3 = LEFT,
        add_content: bool = True,
        fill_color: ManimColor = BLACK,
        fill_opacity: float = 0.8,
        stroke_color: ManimColor = WHITE,
        stroke_width: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.direction = direction

        if content is None:
            content = Rectangle(*filler_shape)
            content.set_fill(opacity=0)
            content.set_stroke(width=0)
        elif isinstance(content, str):
            content = Text(content)
        self.content = content

        self.body = self.get_body(content, direction, buff)
        self.body.set_fill(fill_color, fill_opacity)
        self.body.set_stroke(stroke_color, stroke_width)
        self.add(self.body)

        if add_content:
            self.add(self.content)

        if pin_point is not None:
            self.pin_to(pin_point)

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        body = SVGMobject(self.file_name)
        if direction[0] > 0:
            body.flip()
        # Resize
        width = content.get_width()
        height = content.get_height()
        target_width = width + min(buff, height)
        target_height = 1.35 * (height + buff)  # Magic number?
        body.set_shape(target_width, target_height)
        body.move_to(content)
        body.shift(self.bubble_center_adjustment_factor * body.get_height() * DOWN)
        return body

    def get_tip(self):
        return self.get_corner(DOWN + self.direction)

    def get_bubble_center(self):
        factor = self.bubble_center_adjustment_factor
        return self.get_center() + factor * self.get_height() * UP

    def move_tip_to(self, point):
        self.shift(point - self.get_tip())
        return self

    def flip(self, axis=UP, only_body=True, **kwargs):
        super().flip(axis=axis, **kwargs)
        if only_body:
            # Flip in place, don't use kwargs
            self.content.flip(axis=axis)
        if abs(axis[1]) > 0:
            self.direction = -np.array(self.direction)
        return self

    def pin_to(self, mobject, auto_flip=False):
        mob_center = mobject.get_center()
        want_to_flip = np.sign(mob_center[0]) != np.sign(self.direction[0])
        if want_to_flip and auto_flip:
            self.flip()
        boundary_point = mobject.get_bounding_box_point(UP - self.direction)
        vector_from_center = 1.0 * (boundary_point - mob_center)
        self.move_tip_to(mob_center + vector_from_center)
        return self

    def position_mobject_inside(self, mobject, buff=MED_LARGE_BUFF):
        mobject.set_max_width(self.body.get_width() - 2 * buff)
        mobject.set_max_height(self.body.get_height() / 1.5 - 2 * buff)
        mobject.shift(self.get_bubble_center() - mobject.get_center())
        return mobject

    def add_content(self, mobject):
        self.position_mobject_inside(mobject)
        self.content = mobject
        return self.content

    def write(self, text):
        self.add_content(Text(text))
        return self

    def resize_to_content(self, buff=1.0):  # TODO
        self.body.match_points(self.get_body(
            self.content, self.direction, buff
        ))

    def clear(self):
        self.remove(self.content)
        return self


class SpeechBubble(Bubble):
    def __init__(
        self,
        content: str | VMobject | None = None,
        buff: float = MED_SMALL_BUFF,
        filler_shape: Tuple[float, float] = (2.0, 1.0),
        stem_height_to_bubble_height: float = 0.5,
        stem_top_x_props: Tuple[float, float] = (0.2, 0.3),
        **kwargs
    ):
        self.stem_height_to_bubble_height = stem_height_to_bubble_height
        self.stem_top_x_props = stem_top_x_props
        super().__init__(content, buff, filler_shape, **kwargs)

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        rect = SurroundingRectangle(content, buff=buff)
        rect.round_corners()
        lp = rect.get_corner(DL)
        rp = rect.get_corner(DR)
        stem_height = self.stem_height_to_bubble_height * rect.get_height()
        low_prop, high_prop = self.stem_top_x_props
        triangle = Polygon(
            interpolate(lp, rp, low_prop),
            interpolate(lp, rp, high_prop),
            lp + stem_height * DOWN,
        )
        result = Union(rect, triangle)
        result.insert_n_curves(20)
        if direction[0] > 0:
            result.flip()

        return result


class ThoughtBubble(Bubble):
    def __init__(
        self,
        content: str | VMobject | None = None,
        buff: float = SMALL_BUFF,
        filler_shape: Tuple[float, float] = (2.0, 1.0),
        bulge_radius: float = 0.35,
        bulge_overlap: float = 0.25,
        noise_factor: float = 0.1,
        circle_radii: list[float] = [0.1, 0.15, 0.2],
        **kwargs
    ):
        self.bulge_radius = bulge_radius
        self.bulge_overlap = bulge_overlap
        self.noise_factor = noise_factor
        self.circle_radii = circle_radii
        super().__init__(content, buff, filler_shape, **kwargs)

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        rect = SurroundingRectangle(content, buff)
        perimeter = rect.get_arc_length()
        radius = self.bulge_radius
        step = (1 - self.bulge_overlap) * (2 * radius)
        nf = self.noise_factor
        corners = [rect.get_corner(v) for v in [DL, UL, UR, DR]]
        points = []
        for c1, c2 in adjacent_pairs(corners):
            n_alphas = int(get_norm(c1 - c2) / step) + 1
            for alpha in np.linspace(0, 1, n_alphas):
                points.append(interpolate(
                    c1, c2, alpha + nf * (step / n_alphas) * (random.random() - 0.5)
                ))

        cloud = Union(rect, *(
            # Add bulges
            Circle(radius=radius * (1 + nf * random.random())).move_to(point)
            for point in points
        ))
        cloud.set_stroke(WHITE, 2)

        circles = VGroup(Circle(radius=radius) for radius in self.circle_radii)
        circ_buff = 0.25 * self.circle_radii[0]
        circles.arrange(UR, buff=circ_buff)
        circles[1].shift(circ_buff * DR)
        circles.next_to(cloud, DOWN, 4 * circ_buff, aligned_edge=LEFT)
        circles.set_stroke(WHITE, 2)

        result = VGroup(*circles, cloud)

        if direction[0] > 0:
            result.flip()

        return result


class OldSpeechBubble(Bubble):
    file_name: str = "Bubbles_speech.svg"


class DoubleSpeechBubble(Bubble):
    file_name: str = "Bubbles_double_speech.svg"


class OldThoughtBubble(Bubble):
    file_name: str = "Bubbles_thought.svg"

    def get_body(self, content: VMobject, direction: Vect3, buff: float) -> VMobject:
        body = super().get_body(content, direction, buff)
        body.sort(lambda p: p[1])
        return body

    def make_green_screen(self):
        self.body[-1].set_fill(GREEN_SCREEN, opacity=1)
        return self


class VectorizedEarth(SVGMobject):
    file_name: str = "earth"

    def __init__(
        self,
        height: float = 2.0,
        **kwargs
    ):
        super().__init__(height=height, **kwargs)
        self.insert_n_curves(20)
        circle = Circle(
            stroke_width=3,
            stroke_color=GREEN,
            fill_opacity=1,
            fill_color=BLUE_C,
        )
        circle.replace(self)
        self.add_to_back(circle)


class Piano(VGroup):
    def __init__(
        self,
        n_white_keys = 52,
        black_pattern = [0, 2, 3, 5, 6],
        white_keys_per_octave = 7,
        white_key_dims = (0.15, 1.0),
        black_key_dims = (0.1, 0.66),
        key_buff = 0.02,
        white_key_color = WHITE,
        black_key_color = GREY_E,
        total_width = 13,
        **kwargs
    ):
        self.n_white_keys = n_white_keys
        self.black_pattern = black_pattern
        self.white_keys_per_octave = white_keys_per_octave
        self.white_key_dims = white_key_dims
        self.black_key_dims = black_key_dims
        self.key_buff = key_buff
        self.white_key_color = white_key_color
        self.black_key_color = black_key_color
        self.total_width = total_width

        super().__init__(**kwargs)
        self.add_white_keys()
        self.add_black_keys()
        self.sort_keys()
        self[:-1].reverse_points()
        self.set_width(self.total_width)

    def add_white_keys(self):
        key = Rectangle(*self.white_key_dims)
        key.set_fill(self.white_key_color, 1)
        key.set_stroke(width=0)
        self.white_keys = key.get_grid(1, self.n_white_keys, buff=self.key_buff)
        self.add(*self.white_keys)

    def add_black_keys(self):
        key = Rectangle(*self.black_key_dims)
        key.set_fill(self.black_key_color, 1)
        key.set_stroke(width=0)

        self.black_keys = VGroup()
        for i in range(len(self.white_keys) - 1):
            if i % self.white_keys_per_octave not in self.black_pattern:
                continue
            wk1 = self.white_keys[i]
            wk2 = self.white_keys[i + 1]
            bk = key.copy()
            bk.move_to(midpoint(wk1.get_top(), wk2.get_top()), UP)
            big_bk = bk.copy()
            big_bk.stretch((bk.get_width() + self.key_buff) / bk.get_width(), 0)
            big_bk.stretch((bk.get_height() + self.key_buff) / bk.get_height(), 1)
            big_bk.move_to(bk, UP)
            for wk in wk1, wk2:
                wk.become(Difference(wk, big_bk).match_style(wk))
            self.black_keys.add(bk)
        self.add(*self.black_keys)

    def sort_keys(self):
        self.sort(lambda p: p[0])


class Piano3D(VGroup):
    def __init__(
        self,
        shading: Tuple[float, float, float] = (1.0, 0.2, 0.2),
        stroke_width: float = 0.25,
        stroke_color: ManimColor = BLACK,
        key_depth: float = 0.1,
        black_key_shift: float = 0.05,
        piano_2d_config: dict = dict(
            white_key_color=GREY_A,
            key_buff=0.001
        ),
        **kwargs
    ):
        piano_2d = Piano(**piano_2d_config)
        super().__init__(*(
            Prismify(key, key_depth)
            for key in piano_2d
        ))
        self.set_stroke(stroke_color, stroke_width)
        self.set_shading(*shading)
        self.apply_depth_test()

        # Elevate black keys
        for i, key in enumerate(self):
            if piano_2d[i] in piano_2d.black_keys:
                key.shift(black_key_shift * OUT)
                key.set_color(BLACK)


class DieFace(VGroup):
    def __init__(
        self,
        value: int,
        side_length: float = 1.0,
        corner_radius: float = 0.15,
        stroke_color: ManimColor = WHITE,
        stroke_width: float = 2.0,
        fill_color: ManimColor = GREY_E,
        dot_radius: float = 0.08,
        dot_color: ManimColor = WHITE,
        dot_coalesce_factor: float = 0.5
    ):
        dot = Dot(radius=dot_radius, fill_color=dot_color)
        square = Square(
            side_length=side_length,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=1.0,
        )
        square.round_corners(corner_radius)

        if not (1 <= value <= 6):
            raise Exception("DieFace only accepts integer inputs between 1 and 6")

        edge_group = [
            (ORIGIN,),
            (UL, DR),
            (UL, ORIGIN, DR),
            (UL, UR, DL, DR),
            (UL, UR, ORIGIN, DL, DR),
            (UL, UR, LEFT, RIGHT, DL, DR),
        ][value - 1]

        arrangement = VGroup(*(
            dot.copy().move_to(square.get_bounding_box_point(vect))
            for vect in edge_group
        ))
        arrangement.space_out_submobjects(dot_coalesce_factor)

        super().__init__(square, arrangement)
        self.dots = arrangement
        self.value = value
        self.index = value


class Dartboard(VGroup):
    radius = 3
    n_sectors = 20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        n_sectors = self.n_sectors
        angle = TAU / n_sectors

        segments = VGroup(*[
            VGroup(*[
                AnnularSector(
                    inner_radius=in_r,
                    outer_radius=out_r,
                    start_angle=n * angle,
                    angle=angle,
                    fill_color=color,
                )
                for n, color in zip(
                    range(n_sectors),
                    it.cycle(colors)
                )
            ])
            for colors, in_r, out_r in [
                ([GREY_B, GREY_E], 0, 1),
                ([GREEN_E, RED_E], 0.5, 0.55),
                ([GREEN_E, RED_E], 0.95, 1),
            ]
        ])
        segments.rotate(-angle / 2)
        bullseyes = VGroup(*[
            Circle(radius=r)
            for r in [0.07, 0.035]
        ])
        bullseyes.set_fill(opacity=1)
        bullseyes.set_stroke(width=0)
        bullseyes[0].set_color(GREEN_E)
        bullseyes[1].set_color(RED_E)

        self.bullseye = bullseyes[1]
        self.add(*segments, *bullseyes)
        self.scale(self.radius)
