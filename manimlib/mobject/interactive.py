from __future__ import annotations

import numpy as np

from manimlib.constants import FRAME_HEIGHT, FRAME_WIDTH
from manimlib.constants import DOWN, LEFT, ORIGIN, RIGHT, UP
from manimlib.constants import MED_LARGE_BUFF, MED_SMALL_BUFF, SMALL_BUFF
from manimlib.constants import BLACK, BLUE, GREEN, GREY_A, GREY_C, RED, WHITE, DEFAULT_MOBJECT_COLOR
from manimlib.event_keys import Keys
from manimlib.event_keys import Mods
from manimlib.mobject.mobject import Group
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.geometry import Circle
from manimlib.mobject.geometry import Dot
from manimlib.mobject.geometry import Line
from manimlib.mobject.geometry import Rectangle
from manimlib.mobject.geometry import RoundedRectangle
from manimlib.mobject.geometry import Square
from manimlib.mobject.svg.text_mobject import Text
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.value_tracker import ValueTracker
from manimlib.utils.color import rgb_to_hex
from manimlib.utils.space_ops import get_closest_point_on_line
from manimlib.utils.space_ops import get_norm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.typing import ManimColor


# Interactive Mobjects

class MotionMobject(Mobject):
    """
    Wrap a Mobject to make it draggable with the mouse.

    MotionMobject allows a user to interactively reposition a Mobject by
    clicking and dragging it in the scene. It registers a mouse-drag
    listener on the wrapped object and moves that object to the position
    provided by the drag event.

    The wrapped Mobject receives a no-op updater to help prevent it from
    being treated as a static object.

    Parameters
    ----------
    mobject : Mobject
        The object that should be draggable. The object is added as a
        submobject of the MotionMobject.

    **kwargs
        Additional keyword arguments forwarded to :class:`Mobject`.

    Attributes
    ----------
    mobject : Mobject
        The wrapped object that responds to mouse-drag events.

    Methods
    -------
    mob_on_mouse_drag(mob, event_data)
        Moves the dragged object to the point specified by the event data.

    Examples
    --------
    Make a circle draggable::

        circle = Circle()
        draggable_circle = MotionMobject(circle)
        self.add(draggable_circle)

    Make a text label draggable::

        label = Text("Drag me")
        draggable_label = MotionMobject(label)
        self.add(draggable_label)

    Wrap an existing object without creating a copy::

        square = Square()
        draggable_square = MotionMobject(square)
        self.add(draggable_square)

    Notes
    -----
    - The constructor checks that ``mobject`` is an instance of
      :class:`Mobject`; otherwise, an ``AssertionError`` is raised.
    - The mouse-drag listener is registered on the wrapped object through
      ``add_mouse_drag_listner``.
    - During dragging, ``event_data["point"]`` provides the target position.
    - The callback moves the object using ``mob.move_to(...)`` and returns
      ``False``.
    - The wrapped object is added directly as a submobject, so this wrapper
      does not create an independent copy of it.
    - The no-op updater is attached to the wrapped object to help avoid
      locking it as a static Mobject.
    - Interactive dragging requires a rendering or preview environment
      that supports mouse events.

    See Also
    --------
    Mobject
        Base class for objects in the scene.
    """

    def __init__(self, mobject: Mobject, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(mobject, Mobject)
        self.mobject = mobject
        self.mobject.add_mouse_drag_listner(self.mob_on_mouse_drag)
        # To avoid locking it as static mobject
        self.mobject.add_updater(lambda mob: None)
        self.add(mobject)

    def mob_on_mouse_drag(self, mob: Mobject, event_data: dict[str, np.ndarray]) -> bool:
        mob.move_to(event_data["point"])
        return False


class Button(Mobject):
    """
    Wrap a Mobject and execute a callback when it receives a mouse-press event.

    Button provides a simple way to make an existing Mobject behave like an
    interactive button. It registers a mouse-press listener on the supplied
    Mobject and calls the user-defined ``on_click`` callback when the event
    is dispatched to that object.

    The callback receives the Mobject associated with the mouse-press event,
    similar to how a Mobject updater receives the object it updates.

    Event Handling
    --------------
    The event system follows an event-bubbling model inspired by the DOM
    event model in JavaScript. Mouse and keyboard events are dispatched
    through the event system, and listeners can return ``False`` to stop
    the event from bubbling further.

    Button's internal mouse-press handler calls ``on_click(mob)`` and then
    returns ``False``. This prevents the mouse-press event from continuing
    to bubble beyond this handler according to the event system's
    dispatching rules.

    Parameters
    ----------
    mobject : Mobject
        The visual object that acts as the button. A mouse-press listener
        is registered on this object, which is then added as a submobject
        of the Button.

    on_click : Callable[[Mobject], Any]
        The callback to execute when the button receives a mouse-press
        event. It must accept one argument: the Mobject passed to the
        internal event handler. The callback's return value is not used
        by Button.

    **kwargs
        Additional keyword arguments forwarded to :class:`Mobject`.

    Attributes
    ----------
    on_click : Callable
        The user-provided callback invoked by the mouse-press handler.

    mobject : Mobject
        The visual object wrapped by the Button.

    Methods
    -------
    mob_on_mouse_press(mob, event_data)
        Handles the mouse-press event by invoking ``on_click(mob)`` and
        returning ``False`` to stop further event bubbling.

    Examples
    --------
    Create a button that prints a message when clicked::

        def handle_click(mob):
            print("Button clicked!")

        visual = Square()
        button = Button(visual, on_click=handle_click)
        self.add(button)

    Change an object's color when clicked::

        def change_color(mob):
            mob.set_color(RED)

        square = Square()
        button = Button(square, on_click=change_color)
        self.add(button)

    Use a text object as a button::

        def handle_click(mob):
            print("Selected:", mob)

        label = Text("Click me")
        button = Button(label, on_click=handle_click)
        self.add(button)

    Modify the scene from a callback::

        def enlarge(mob):
            mob.scale(1.2)

        circle = Circle()
        button = Button(circle, on_click=enlarge)
        self.add(button)

    Notes
    -----
    - ``mobject`` must be an instance of :class:`Mobject`; otherwise, the
      constructor raises an ``AssertionError``.
    - The supplied Mobject is used directly rather than copied.
    - The callback receives one argument, the event-associated Mobject.
      It does not receive ``event_data`` through the Button callback.
    - The internal handler accepts ``event_data`` because it follows the
      event-listener callback interface, but does not use it.
    - The mouse-press listener is registered through
      ``add_mouse_press_listner``.
    - Button does not itself perform hit-testing, draw a button background,
      or define hover, pressed, or released visual states. Those behaviors
      depend on the event system and any additional code.
    - The interaction requires a rendering or preview environment that
      dispatches mouse events.
    - The event system also exposes listeners for mouse motion, press,
      release, drag, scroll, and keyboard press and release events.
      These can be registered or removed using the corresponding
      Mobject listener methods.

    See Also
    --------
    Mobject
        Base class for scene objects and event-listener registration.
    MotionMobject
        Wraps a Mobject to make it draggable using mouse-drag events.

    References
    ----------
    Event bubbling in the DOM event model:
    https://www.quirksmode.org/js/events_order.html
    """

    def __init__(self, mobject: Mobject, on_click: Callable[[Mobject]], **kwargs):
        super().__init__(**kwargs)
        assert isinstance(mobject, Mobject)
        self.on_click = on_click
        self.mobject = mobject
        self.mobject.add_mouse_press_listner(self.mob_on_mouse_press)
        self.add(self.mobject)

    def mob_on_mouse_press(self, mob: Mobject, event_data) -> bool:
        self.on_click(mob)
        return False


# Controls

class ControlMobject(ValueTracker):
    """
    A ValueTracker-based base class for interactive controls in a scene.

    ControlMobject extends ValueTracker with support for attaching visual
    Mobjects and defining custom behavior when its value changes. It is
    intended to be subclassed to implement controls that validate a new
    value and animate the visual representation of that value.

    The control is fixed in the camera frame and receives a no-op updater
    to help prevent its data from being locked as static while waiting
    in a scene.

    Parameters
    ----------
    value : float
        The initial value stored by the ValueTracker.

    *mobjects : Mobject
        Zero or more Mobjects to attach to the control. These can represent
        the control's visual components.

    **kwargs
        Additional keyword arguments forwarded to :class:`ValueTracker`.

    Attributes
    ----------
    Inherited from ValueTracker
        The tracked numeric value and the methods used to access or update it.

    Methods
    -------
    set_value(value)
        Validates the requested value, invokes the animation hook, and then
        updates the underlying ValueTracker value.

    assert_value(value)
        Validation hook for subclasses. The base implementation does nothing.

    set_value_anim(value)
        Animation hook for subclasses. The base implementation does nothing.

    Examples
    --------
    Create a basic control with an initial value::

        control = ControlMobject(0.5)

    Attach visual objects to a control::

        track = Line(LEFT, RIGHT)
        marker = Dot()

        control = ControlMobject(0.5, track, marker)
        self.add(control)

    Subclass ControlMobject to validate values and define custom animation::

        class BoundedControl(ControlMobject):
            def assert_value(self, value):
                assert 0 <= value <= 1, "Value must be between 0 and 1"

            def set_value_anim(self, value):
                # Implement the visual update or animation here.
                pass

        control = BoundedControl(0.5)

    Notes
    -----
    - ``ControlMobject`` is a base class intended for extension. Its
      ``assert_value`` and ``set_value_anim`` methods are placeholders and
      do not implement validation or animation by themselves.
    - ``set_value`` calls ``assert_value(value)`` first, followed by
      ``set_value_anim(value)``, and then delegates the actual value update
      to ``ValueTracker.set_value(self, value)``.
    - If validation raises an exception, the animation hook and underlying
      value update are not reached.
    - Subclasses should implement ``assert_value`` to enforce valid input
      and ``set_value_anim`` to update or animate their visual components.
    - The constructor attaches the supplied Mobjects as submobjects.
    - The no-op updater is added to help avoid static-mobject data locking.
    - ``fix_in_frame()`` keeps the control fixed relative to the camera
      frame rather than behaving like an ordinary world-space object.
    - The class does not define a particular visual appearance or input
      mechanism. Subclasses must provide the behavior appropriate to
      their intended control.

    See Also
    --------
    ValueTracker
        Stores a numeric value that can be accessed and updated.
    Mobject
        Base class for visual objects in a scene.
    """

    def __init__(self, value: float, *mobjects: Mobject, **kwargs):
        super().__init__(value=value, **kwargs)
        self.add(*mobjects)

        # To avoid lock_static_mobject_data while waiting in scene
        self.add_updater(lambda mob: None)
        self.fix_in_frame()

    def set_value(self, value: float):
        self.assert_value(value)
        self.set_value_anim(value)
        return ValueTracker.set_value(self, value)

    def assert_value(self, value):
        # To be implemented in subclasses
        pass

    def set_value_anim(self, value):
        # To be implemented in subclasses
        pass


class EnableDisableButton(ControlMobject):
    """
    A clickable Boolean control that switches between enabled and disabled states.

    EnableDisableButton is a ControlMobject subclass that represents a Boolean
    value using a colored rectangle. It registers a mouse-press listener so
    that clicking the control toggles its state.

    When enabled, the rectangle uses ``enable_color``; when disabled, it uses
    ``disable_color``. The current value is stored through the inherited
    ValueTracker interface.

    Parameters
    ----------
    value : bool, default=True
        Initial state of the control. ``True`` represents enabled, and
        ``False`` represents disabled.

    value_type : np.dtype, default=np.dtype(bool)
        NumPy data type associated with the control's value. The constructor
        stores this parameter, but the shown implementation does not use it
        to convert or validate values.

    rect_kwargs : dict, default={"width": 0.5, "height": 0.5, "fill_opacity": 1.0}
        Keyword arguments passed to Rectangle to configure the control's
        visual appearance.

    enable_color : ManimColor, default=GREEN
        Fill color used when the control is enabled.

    disable_color : ManimColor, default=RED
        Fill color used when the control is disabled.

    **kwargs
        Additional keyword arguments forwarded to ControlMobject.

    Attributes
    ----------
    value : bool
        Stores the initial value supplied to the constructor. In this
        implementation, this attribute is not automatically updated when
        the inherited ValueTracker value changes.

    value_type : np.dtype
        The stored NumPy data type parameter.

    rect_kwargs : dict
        Configuration passed to the Rectangle constructor.

    enable_color : ManimColor
        Color used for the enabled state.

    disable_color : ManimColor
        Color used for the disabled state.

    box : Rectangle
        Rectangle that visually represents the control's current state.

    Methods
    -------
    assert_value(value)
        Checks that the supplied value is a Python bool.

    set_value_anim(value)
        Updates the rectangle's fill color to reflect the requested state.

    toggle_value()
        Switches the tracked value between True and False.

    on_mouse_press(mob, event_data)
        Toggles the control when a mouse-press event is received.

    Examples
    --------
    Create an enabled control::

        control = EnableDisableButton()
        self.add(control)

    Create a control that starts disabled::

        control = EnableDisableButton(value=False)
        self.add(control)

    Customize the enabled and disabled colors::

        control = EnableDisableButton(
            value=True,
            enable_color=BLUE,
            disable_color=GRAY,
        )
        self.add(control)

    Toggle the state programmatically::

        control = EnableDisableButton(value=True)
        control.toggle_value()

        print(control.get_value())  # False

    Set the state explicitly::

        control = EnableDisableButton()
        control.set_value(False)

    Notes
    -----
    - The control creates a Rectangle using ``rect_kwargs`` and passes it
      to the ControlMobject constructor as a visual submobject.
    - ``assert_value`` uses ``isinstance(value, bool)``. Values such as
      ``1`` and ``np.bool_(True)`` are not accepted by this check.
    - ``set_value_anim`` changes the rectangle's fill color immediately;
      it does not define a time-based animation.
    - ``toggle_value`` obtains the current tracked value, negates it, and
      calls the parent class's ``set_value`` method. This invokes the
      validation and visual-update hooks before updating the ValueTracker.
    - A mouse press invokes ``toggle_value`` and returns ``False`` from
      the event callback to stop further event bubbling according to the
      event system's dispatch rules.
    - The constructor stores the initial value in ``self.value`` before
      initializing the parent class. Subsequent changes update the inherited
      tracked value; the separate ``self.value`` attribute is not explicitly
      synchronized by the shown methods.
    - ``value_type`` is stored for possible use by subclasses or other code,
      but it does not perform type conversion in this implementation.
    - Interactive clicking requires an environment that dispatches mouse
      events to Mobjects.

    See Also
    --------
    ControlMobject
        Base class for value-based visual controls.
    Button
        Wraps a Mobject and invokes a callback on mouse press.
    ValueTracker
        Stores a value that can be read and updated.
    """

    def __init__(
        self,
        value: bool = True,
        value_type: np.dtype = np.dtype(bool),
        rect_kwargs: dict = {
            "width": 0.5,
            "height": 0.5,
            "fill_opacity": 1.0
        },
        enable_color: ManimColor = GREEN,
        disable_color: ManimColor = RED,
        **kwargs
    ):
        self.value = value
        self.value_type = value_type
        self.rect_kwargs = rect_kwargs
        self.enable_color = enable_color
        self.disable_color = disable_color

        self.box = Rectangle(**self.rect_kwargs)
        super().__init__(value, self.box, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value: bool) -> None:
        assert isinstance(value, bool)

    def set_value_anim(self, value: bool) -> None:
        if value:
            self.box.set_fill(self.enable_color)
        else:
            self.box.set_fill(self.disable_color)

    def toggle_value(self) -> None:
        super().set_value(not self.get_value())

    def on_mouse_press(self, mob: Mobject, event_data) -> bool:
        mob.toggle_value()
        return False


class Checkbox(ControlMobject):
    def __init__(
        self,
        value: bool = True,
        value_type: np.dtype = np.dtype(bool),
        rect_kwargs: dict = {
            "width": 0.5,
            "height": 0.5,
            "fill_opacity": 0.0
        },
        checkmark_kwargs: dict = {
            "stroke_color": GREEN,
            "stroke_width": 6,
        },
        cross_kwargs: dict = {
            "stroke_color": RED,
            "stroke_width": 6,
        },
        box_content_buff: float = SMALL_BUFF,
        **kwargs
    ):
        self.value_type = value_type
        self.rect_kwargs = rect_kwargs
        self.checkmark_kwargs = checkmark_kwargs
        self.cross_kwargs = cross_kwargs
        self.box_content_buff = box_content_buff

        self.box = Rectangle(**self.rect_kwargs)
        self.box_content = self.get_checkmark() if value else self.get_cross()
        super().__init__(value, self.box, self.box_content, **kwargs)
        self.add_mouse_press_listner(self.on_mouse_press)

    def assert_value(self, value: bool) -> None:
        assert isinstance(value, bool)

    def toggle_value(self) -> None:
        super().set_value(not self.get_value())

    def set_value_anim(self, value: bool) -> None:
        if value:
            self.box_content.become(self.get_checkmark())
        else:
            self.box_content.become(self.get_cross())

    def on_mouse_press(self, mob: Mobject, event_data) -> None:
        mob.toggle_value()
        return False

    # Helper methods

    def get_checkmark(self) -> VGroup:
        checkmark = VGroup(
            Line(UP / 2 + 2 * LEFT, DOWN + LEFT, **self.checkmark_kwargs),
            Line(DOWN + LEFT, UP + RIGHT, **self.checkmark_kwargs)
        )

        checkmark.stretch_to_fit_width(self.box.get_width())
        checkmark.stretch_to_fit_height(self.box.get_height())
        checkmark.scale(0.5)
        checkmark.move_to(self.box)
        return checkmark

    def get_cross(self) -> VGroup:
        cross = VGroup(
            Line(UP + LEFT, DOWN + RIGHT, **self.cross_kwargs),
            Line(UP + RIGHT, DOWN + LEFT, **self.cross_kwargs)
        )

        cross.stretch_to_fit_width(self.box.get_width())
        cross.stretch_to_fit_height(self.box.get_height())
        cross.scale(0.5)
        cross.move_to(self.box)
        return cross


class LinearNumberSlider(ControlMobject):
    def __init__(
        self,
        value: float = 0,
        value_type: type = np.float64,
        min_value: float = -10.0,
        max_value: float = 10.0,
        step: float = 1.0,
        rounded_rect_kwargs: dict = {
            "height": 0.075,
            "width": 2,
            "corner_radius": 0.0375
        },
        circle_kwargs: dict = {
            "radius": 0.1,
            "stroke_color": GREY_A,
            "fill_color": GREY_A,
            "fill_opacity": 1.0
        },
        **kwargs
    ):
        self.value_type = value_type
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.rounded_rect_kwargs = rounded_rect_kwargs
        self.circle_kwargs = circle_kwargs

        self.bar = RoundedRectangle(**self.rounded_rect_kwargs)
        self.slider = Circle(**self.circle_kwargs)
        self.slider_axis = Line(
            start=self.bar.get_bounding_box_point(LEFT),
            end=self.bar.get_bounding_box_point(RIGHT)
        )
        self.slider_axis.set_opacity(0.0)
        self.slider.move_to(self.slider_axis)

        self.slider.add_mouse_drag_listner(self.slider_on_mouse_drag)

        super().__init__(value, self.bar, self.slider, self.slider_axis, **kwargs)

    def assert_value(self, value: float) -> None:
        assert self.min_value <= value <= self.max_value

    def set_value_anim(self, value: float) -> None:
        prop = (value - self.min_value) / (self.max_value - self.min_value)
        self.slider.move_to(self.slider_axis.point_from_proportion(prop))

    def slider_on_mouse_drag(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        self.set_value(self.get_value_from_point(event_data["point"]))
        return False

    # Helper Methods

    def get_value_from_point(self, point: np.ndarray) -> float:
        start, end = self.slider_axis.get_start_and_end()
        point_on_line = get_closest_point_on_line(start, end, point)
        prop = get_norm(point_on_line - start) / get_norm(end - start)
        value = self.min_value + prop * (self.max_value - self.min_value)
        no_of_steps = int((value - self.min_value) / self.step)
        value_nearest_to_step = self.min_value + no_of_steps * self.step
        return value_nearest_to_step


class ColorSliders(Group):
    def __init__(
        self,
        sliders_kwargs: dict = {},
        rect_kwargs: dict = {
            "width": 2.0,
            "height": 0.5,
            "stroke_opacity": 1.0
        },
        background_grid_kwargs: dict = {
            "colors": [GREY_A, GREY_C],
            "single_square_len": 0.1
        },
        sliders_buff: float = MED_LARGE_BUFF,
        default_rgb_value: int = 255,
        default_a_value: int = 1,
        **kwargs
    ):
        self.sliders_kwargs = sliders_kwargs
        self.rect_kwargs = rect_kwargs
        self.background_grid_kwargs = background_grid_kwargs
        self.sliders_buff = sliders_buff
        self.default_rgb_value = default_rgb_value
        self.default_a_value = default_a_value

        rgb_kwargs = {"value": self.default_rgb_value, "min_value": 0, "max_value": 255, "step": 1}
        a_kwargs = {"value": self.default_a_value, "min_value": 0, "max_value": 1, "step": 0.04}

        self.r_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)
        self.g_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)
        self.b_slider = LinearNumberSlider(**self.sliders_kwargs, **rgb_kwargs)
        self.a_slider = LinearNumberSlider(**self.sliders_kwargs, **a_kwargs)
        self.sliders = Group(
            self.r_slider,
            self.g_slider,
            self.b_slider,
            self.a_slider
        )
        self.sliders.arrange(DOWN, buff=self.sliders_buff)

        self.r_slider.slider.set_color(RED)
        self.g_slider.slider.set_color(GREEN)
        self.b_slider.slider.set_color(BLUE)
        self.a_slider.slider.set_color_by_gradient(BLACK, WHITE)

        self.selected_color_box = Rectangle(**self.rect_kwargs)
        self.selected_color_box.add_updater(
            lambda mob: mob.set_fill(
                self.get_picked_color(), self.get_picked_opacity()
            )
        )
        self.background = self.get_background()

        super().__init__(
            Group(self.background, self.selected_color_box).fix_in_frame(),
            self.sliders,
            **kwargs
        )

        self.arrange(DOWN)

    def get_background(self) -> VGroup:
        single_square_len = self.background_grid_kwargs["single_square_len"]
        colors = self.background_grid_kwargs["colors"]
        width = self.rect_kwargs["width"]
        height = self.rect_kwargs["height"]
        rows = int(height / single_square_len)
        cols = int(width / single_square_len)
        cols = (cols + 1) if (cols % 2 == 0) else cols

        single_square = Square(single_square_len)
        grid = single_square.get_grid(n_rows=rows, n_cols=cols, buff=0.0)
        grid.stretch_to_fit_width(width)
        grid.stretch_to_fit_height(height)
        grid.move_to(self.selected_color_box)

        for idx, square in enumerate(grid):
            assert isinstance(square, Square)
            square.set_stroke(width=0.0, opacity=0.0)
            square.set_fill(colors[idx % len(colors)], 1.0)

        return grid

    def set_value(self, r: float, g: float, b: float, a: float):
        self.r_slider.set_value(r)
        self.g_slider.set_value(g)
        self.b_slider.set_value(b)
        self.a_slider.set_value(a)

    def get_value(self) -> np.ndarary:
        r = self.r_slider.get_value() / 255
        g = self.g_slider.get_value() / 255
        b = self.b_slider.get_value() / 255
        alpha = self.a_slider.get_value()
        return np.array((r, g, b, alpha))

    def get_picked_color(self) -> str:
        rgba = self.get_value()
        return rgb_to_hex(rgba[:3])

    def get_picked_opacity(self) -> float:
        rgba = self.get_value()
        return rgba[3]


class Textbox(ControlMobject):
    def __init__(
        self,
        value: str = "",
        value_type: np.dtype = np.dtype(object),
        box_kwargs: dict = {
            "width": 2.0,
            "height": 1.0,
            "fill_color": DEFAULT_MOBJECT_COLOR,
            "fill_opacity": 1.0,
        },
        text_kwargs: dict = {
            "color": BLUE
        },
        text_buff: float = MED_SMALL_BUFF,
        isInitiallyActive: bool = False,
        active_color: ManimColor = BLUE,
        deactive_color: ManimColor = RED,
        **kwargs
    ):
        self.value_type = value_type
        self.box_kwargs = box_kwargs
        self.text_kwargs = text_kwargs
        self.text_buff = text_buff
        self.isInitiallyActive = isInitiallyActive
        self.active_color = active_color
        self.deactive_color = deactive_color

        self.isActive = self.isInitiallyActive
        self.box = Rectangle(**self.box_kwargs)
        self.box.add_mouse_press_listner(self.box_on_mouse_press)
        self.text = Text(value, **self.text_kwargs)
        super().__init__(value, self.box, self.text, **kwargs)
        self.update_text(value)
        self.active_anim(self.isActive)
        self.add_key_press_listner(self.on_key_press)

    def set_value_anim(self, value: str) -> None:
        self.update_text(value)

    def update_text(self, value: str) -> None:
        text = self.text
        self.remove(text)
        text.__init__(value, **self.text_kwargs)
        height = text.get_height()
        text.set_width(self.box.get_width() - 2 * self.text_buff)
        if text.get_height() > height:
            text.set_height(height)
        text.add_updater(lambda mob: mob.move_to(self.box))
        text.fix_in_frame()
        self.add(text)

    def active_anim(self, isActive: bool) -> None:
        if isActive:
            self.box.set_stroke(self.active_color)
        else:
            self.box.set_stroke(self.deactive_color)

    def box_on_mouse_press(self, mob, event_data) -> bool:
        self.isActive = not self.isActive
        self.active_anim(self.isActive)
        return False

    def on_key_press(self, mob: Mobject, event_data: dict[str, int]) -> bool | None:
        symbol = event_data["symbol"]
        modifiers = event_data["modifiers"]
        char = chr(symbol)
        if mob.isActive:
            old_value = mob.get_value()
            new_value = old_value
            if char.isalnum():
                if modifiers & Mods.SHIFT:
                    new_value = old_value + char.upper()
                else:
                    new_value = old_value + char.lower()
            elif symbol == Keys.SPACE:
                new_value = old_value + char
            elif symbol == Keys.TAB:
                new_value = old_value + '\t'
            elif symbol == Keys.BACKSPACE:
                new_value = old_value[:-1] or ''
            mob.set_value(new_value)
            return False


class ControlPanel(Group):
    def __init__(
        self,
        *controls: ControlMobject,
        panel_kwargs: dict = {
            "width": FRAME_WIDTH / 4,
            "height": MED_SMALL_BUFF + FRAME_HEIGHT,
            "fill_color": GREY_C,
            "fill_opacity": 1.0,
            "stroke_width": 0.0
        },
        opener_kwargs: dict = {
            "width": FRAME_WIDTH / 8,
            "height": 0.5,
            "fill_color": GREY_C,
            "fill_opacity": 1.0
        },
        opener_text_kwargs: dict = {
            "text": "Control Panel",
            "font_size": 20
        },
        **kwargs
    ):
        self.panel_kwargs = panel_kwargs
        self.opener_kwargs = opener_kwargs
        self.opener_text_kwargs = opener_text_kwargs

        self.panel = Rectangle(**self.panel_kwargs)
        self.panel.to_corner(UP + LEFT, buff=0)
        self.panel.shift(self.panel.get_height() * UP)
        self.panel.add_mouse_scroll_listner(self.panel_on_mouse_scroll)

        self.panel_opener_rect = Rectangle(**self.opener_kwargs)
        self.panel_info_text = Text(**self.opener_text_kwargs)
        self.panel_info_text.move_to(self.panel_opener_rect)

        self.panel_opener = Group(self.panel_opener_rect, self.panel_info_text)
        self.panel_opener.next_to(self.panel, DOWN, aligned_edge=DOWN)
        self.panel_opener.add_mouse_drag_listner(self.panel_opener_on_mouse_drag)

        self.controls = Group(*controls)
        self.controls.arrange(DOWN, center=False, aligned_edge=ORIGIN)
        self.controls.move_to(self.panel)

        super().__init__(
            self.panel, self.panel_opener,
            self.controls,
            **kwargs
        )

        self.move_panel_and_controls_to_panel_opener()
        self.fix_in_frame()

    def move_panel_and_controls_to_panel_opener(self) -> None:
        self.panel.next_to(
            self.panel_opener_rect,
            direction=UP,
            buff=0
        )

        controls_old_x = self.controls.get_x()
        self.controls.next_to(
            self.panel_opener_rect,
            direction=UP,
            buff=MED_SMALL_BUFF
        )

        self.controls.set_x(controls_old_x)

    def add_controls(self, *new_controls: ControlMobject) -> None:
        self.controls.add(*new_controls)
        self.move_panel_and_controls_to_panel_opener()

    def remove_controls(self, *controls_to_remove: ControlMobject) -> None:
        self.controls.remove(*controls_to_remove)
        self.move_panel_and_controls_to_panel_opener()

    def open_panel(self):
        panel_opener_x = self.panel_opener.get_x()
        self.panel_opener.to_corner(DOWN + LEFT, buff=0.0)
        self.panel_opener.set_x(panel_opener_x)
        self.move_panel_and_controls_to_panel_opener()
        return self

    def close_panel(self):
        panel_opener_x = self.panel_opener.get_x()
        self.panel_opener.to_corner(UP + LEFT, buff=0.0)
        self.panel_opener.set_x(panel_opener_x)
        self.move_panel_and_controls_to_panel_opener()
        return self

    def panel_opener_on_mouse_drag(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        point = event_data["point"]
        self.panel_opener.match_y(Dot(point))
        self.move_panel_and_controls_to_panel_opener()
        return False

    def panel_on_mouse_scroll(self, mob, event_data: dict[str, np.ndarray]) -> bool:
        offset = event_data["offset"]
        factor = 10 * offset[1]
        self.controls.set_y(self.controls.get_y() + factor)
        return False
