from __future__ import annotations

import numpy as np

from manimlib.animation.animation import Animation
from manimlib.animation.transform import Transform
from manimlib.constants import ORIGIN
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.mobject.mobject import Group
from manimlib.utils.bezier import interpolate
from manimlib.utils.rate_functions import there_and_back

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable
    from manimlib.mobject.mobject import Mobject
    from manimlib.scene.scene import Scene
    from manimlib.typing import Vect3


class Fade(Transform):
    """
    Animates a Mobject by transforming it while applying a positional shift and
    a scale factor.

    Fade extends Transform and stores a shift vector and scale factor before
    forwarding the Mobject and additional arguments to Transform. The exact
    transformation behavior depends on how the parent Transform class uses these
    attributes.

    Parameters
    ----------
    mobject : Mobject
        The object to animate.

    shift : np.ndarray, optional
        Vector describing the positional shift associated with the animation.
        Defaults to ORIGIN, meaning a zero displacement.

    scale : float, optional
        Scale factor associated with the animation. Defaults to 1, meaning the
        original scale.

    **kwargs
        Additional keyword arguments forwarded to Transform.

    Attributes
    ----------
    shift_vect : np.ndarray
        Stores the supplied shift vector.

    scale_factor : float
        Stores the supplied scale factor.

    Notes
    -----
    - Fade inherits its transformation lifecycle and interpolation behavior from
      Transform.
    - The constructor stores shift in self.shift_vect and scale in
      self.scale_factor.
    - This class does not directly modify the Mobject's position or scale in the
      supplied implementation; it passes the Mobject to Transform after storing
      these values.
    - The actual visual result depends on the parent Transform implementation.

    Examples
    --------
    Example 1: Create a Fade animation with default values.

        >>> fade = Fade(circle)

    Here, circle is the Mobject being animated. Since shift defaults to ORIGIN
    and scale defaults to 1, no additional displacement or scaling is requested
    through these parameters.

    Example 2: Configure a shift to the right.

        >>> fade = Fade(circle, shift=RIGHT)

    The shift vector is stored in fade.shift_vect. How it affects the animation
    depends on the Transform implementation.

    Example 3: Configure a scale factor of 0.5.

        >>> fade = Fade(circle, scale=0.5)

    The scale factor is stored in fade.scale_factor. A value of 0.5 represents
    half the original scale if the parent transformation uses this attribute
    as a scale factor.

    Example 4: Combine a shift and a scale factor.

        >>> fade = Fade(
        ...     circle,
        ...     shift=UP,
        ...     scale=0.5,
        ...     run_time=2,
        ... )

    This constructs the animation with both parameters and requests a two-second
    runtime from the parent class.

    In these examples, circle must be a defined Mobject, and RIGHT and UP must
    be available direction vectors.

    See Also
    --------
    Transform
    Mobject
    """

    def __init__(
        self,
        mobject: Mobject,
        shift: np.ndarray = ORIGIN,
        scale: float = 1,
        **kwargs
    ):
        self.shift_vect = shift
        self.scale_factor = scale
        super().__init__(mobject, **kwargs)


class FadeIn(Fade):
    """
    Animates a Mobject into view by transitioning it from a modified starting
    state to its original appearance.

    FadeIn extends Fade. It creates a copy of the original Mobject as the target,
    then constructs a starting Mobject that is fully transparent, inversely
    scaled by the configured scale factor, and shifted in the opposite direction
    to the configured shift vector.

    The animation interpolates from this starting state toward the target using
    the transformation behavior inherited from Transform.

    Parameters
    ----------
    mobject : Mobject
        The object to animate into view.

    shift : np.ndarray, optional
        Vector describing the direction and distance from which the object
        starts moving toward its target. Defaults to ORIGIN, so no positional
        shift is applied.

    scale : float, optional
        Scale factor used to determine the starting object's size. Defaults to 1.
        The starting Mobject is scaled by 1.0 / scale.

    **kwargs
        Additional keyword arguments forwarded through Fade to Transform.

    Attributes
    ----------
    shift_vect : np.ndarray
        Inherited from Fade. Stores the requested shift vector.

    scale_factor : float
        Inherited from Fade. Stores the scale factor used to construct the
        starting Mobject.

    Methods
    -------
    create_target()
        Returns a copy of the original Mobject. This copy represents the target
        appearance of the animation.

    create_starting_mobject()
        Creates the starting state by calling the parent implementation,
        setting its opacity to zero, scaling it by the reciprocal of
        scale_factor, and shifting it by the negative of shift_vect.

    Notes
    -----
    - The target is a copy of the original Mobject.
    - The starting Mobject has zero opacity.
    - The starting scale is determined by 1.0 / scale_factor.
    - The starting position is offset by -shift_vect.
    - The actual transition between the starting state and target is handled by
      the inherited Transform implementation.
    - A scale_factor of zero causes division by zero when the starting Mobject
      is constructed, so it should not be used.

    Examples
    --------
    Example 1: Fade in a circle with default settings.

        >>> animation = FadeIn(circle)

    The circle starts transparent and transitions toward its original appearance.
    Because the default shift is ORIGIN and scale is 1, no additional starting
    offset or size change is applied.

    Example 2: Fade in while moving upward.

        >>> animation = FadeIn(circle, shift=UP)

    The starting Mobject is shifted by -UP, so it begins below its target and
    moves toward the original position during the transformation.

    Example 3: Fade in while scaling up.

        >>> animation = FadeIn(circle, scale=0.5)

    The starting Mobject is scaled by 1 / 0.5 = 2, so it starts at twice the
    target's scale and transitions toward the target.

    Example 4: Combine shifting and scaling.

        >>> animation = FadeIn(
        ...     circle,
        ...     shift=RIGHT,
        ...     scale=0.5,
        ...     run_time=2,
        ... )

    The starting Mobject is transparent, shifted left by the RIGHT vector, and
    scaled to twice its target size. It then transitions toward the target over
    the requested runtime.

    These examples illustrate construction only; circle must be a defined
    Mobject, and direction vectors such as UP and RIGHT must be available.

    See Also
    --------
    Fade
    FadeOut
    Transform
    Mobject
    """

    def create_target(self) -> Mobject:
        return self.mobject.copy()

    def create_starting_mobject(self) -> Mobject:
        start = super().create_starting_mobject()
        start.set_opacity(0)
        start.scale(1.0 / self.scale_factor)
        start.shift(-self.shift_vect)
        return start


class FadeOut(Fade):
    """
    Animates a Mobject out of view by transitioning it toward a transparent,
    shifted, and scaled target state.

    FadeOut extends Fade. It creates a copy of the original Mobject, makes the
    copy fully transparent, shifts it by the configured shift vector, and scales
    it by the configured scale factor. The inherited Transform behavior then
    interpolates the original Mobject toward this target state.

    Parameters
    ----------
    mobject : Mobject
        The object to animate out of view.

    shift : Vect3, optional
        Vector describing the displacement applied to the target Mobject.
        Defaults to ORIGIN, meaning no positional shift.

    remover : bool, optional
        Whether the animation should remove the Mobject from the scene when it
        finishes. Defaults to True.

    final_alpha_value : float, optional
        Final alpha value used by the parent animation implementation.
        Defaults to 0.0. The supplied code forwards this value to Fade/Transform.

    **kwargs
        Additional keyword arguments forwarded through Fade to Transform.

    Attributes
    ----------
    shift_vect : np.ndarray
        Inherited from Fade. Stores the shift vector applied to the target.

    scale_factor : float
        Inherited from Fade. Stores the scale factor applied to the target.

    Methods
    -------
    create_target()
        Creates a copy of the original Mobject, sets its opacity to zero, shifts
        it by shift_vect, scales it by scale_factor, and returns the resulting
        target Mobject.

    Notes
    -----
    - The original Mobject is copied before its target state is modified.
    - The target's opacity is set to zero, making it fully transparent.
    - The target is shifted by shift_vect.
    - The target is scaled by scale_factor.
    - The actual transition from the original state to the target is handled by
      the inherited Transform implementation.
    - With the default remover=True, the animation is configured to remove the
      Mobject from the scene when it finishes.
    - Setting remover=False disables that automatic removal behavior.
    - The scale parameter defaults to 1 in Fade, so FadeOut does not change the
      target's scale unless a different value is supplied.
    - final_alpha_value is passed to the parent implementation; its precise
      effect depends on the implementation of Fade and Transform.

    Examples
    --------
    Example 1: Fade out a circle without moving or scaling it.

        >>> animation = FadeOut(circle)

    The target is a transparent copy at the same position and scale. By default,
    the original Mobject is removed from the scene when the animation finishes.

    Example 2: Fade out while moving to the right.

        >>> animation = FadeOut(circle, shift=RIGHT)

    The target copy is shifted by RIGHT while becoming transparent.

    Example 3: Fade out while shrinking.

        >>> animation = FadeOut(circle, scale=0.5)

    The target copy is scaled to half its original size and made transparent.

    Example 4: Keep the Mobject in the scene after the animation.

        >>> animation = FadeOut(circle, remover=False)

    The animation is configured not to remove the Mobject automatically when it
    finishes.

    Example 5: Combine displacement and scaling.

        >>> animation = FadeOut(
        ...     circle,
        ...     shift=UP,
        ...     scale=0.5,
        ...     run_time=2,
        ... )

    The target is transparent, shifted upward, and scaled to half its original
    size.

    These examples illustrate construction only; circle must be a defined
    Mobject, and direction vectors such as RIGHT and UP must be available.

    See Also
    --------
    Fade
    FadeIn
    Transform
    Mobject
    """

    def __init__(
        self,
        mobject: Mobject,
        shift: Vect3 = ORIGIN,
        remover: bool = True,
        final_alpha_value: float = 0.0,  # Put it back in original state when done,
        **kwargs
    ):
        super().__init__(
            mobject, shift,
            remover=remover,
            final_alpha_value=final_alpha_value,
            **kwargs
        )

    def create_target(self) -> Mobject:
        result = self.mobject.copy()
        result.set_opacity(0)
        result.shift(self.shift_vect)
        result.scale(self.scale_factor)
        return result


class FadeInFromPoint(FadeIn):
    """
    Animates a Mobject into view from a specified point.

    FadeInFromPoint extends FadeIn and configures the starting state so that
    the Mobject begins at the supplied point and transitions toward its original
    position. It calculates the shift vector as the difference between the
    Mobject's center and the specified point, then passes an infinite scale
    factor to FadeIn.

    Parameters
    ----------
    mobject : Mobject
        The object to animate into view.

    point : Vect3
        The point from which the object should appear. The shift vector is
        calculated as mobject.get_center() - point.

    **kwargs
        Additional keyword arguments forwarded to FadeIn, such as run_time
        or rate_func.

    Notes
    -----
    - The class inherits its animation behavior from FadeIn and Fade.
    - The shift is calculated automatically; callers do not supply shift
      directly to this constructor.
    - FadeIn's starting Mobject is shifted by the negative of the calculated
      shift vector. This places its starting center at the specified point.
    - The scale factor is set to np.inf. In FadeIn.create_starting_mobject(),
      the starting Mobject is scaled by 1.0 / scale_factor, which evaluates to
      zero for an infinite scale factor.
    - A zero scale factor can be problematic for some Mobject implementations.
      The exact visual result depends on how the underlying scaling operation
      handles a scale of zero.
    - This class does not define its own target or interpolation methods.

    Examples
    --------
    Example 1: Make a circle appear from the origin.

        >>> animation = FadeInFromPoint(circle, ORIGIN)

    The animation calculates the displacement from the origin to the circle's
    center and uses that to configure the starting state.

    Example 2: Make a square appear from a point above it.

        >>> animation = FadeInFromPoint(square, 2 * UP)

    The square's starting position is configured around the point 2 * UP, then
    the inherited FadeIn animation transitions toward the original square.

    Example 3: Set a custom runtime.

        >>> animation = FadeInFromPoint(
        ...     circle,
        ...     point=LEFT,
        ...     run_time=2,
        ... )

    The animation is configured to begin from LEFT and use a two-second runtime.

    These examples illustrate construction only; circle, square, and direction
    vectors must be defined in the surrounding scene.

    See Also
    --------
    FadeIn
    FadeOut
    Fade
    Transform
    Mobject
    """

    def __init__(self, mobject: Mobject, point: Vect3, **kwargs):
        super().__init__(
            mobject,
            shift=mobject.get_center() - point,
            scale=np.inf,
            **kwargs,
        )


class FadeOutToPoint(FadeOut):
    def __init__(self, mobject: Mobject, point: Vect3, **kwargs):
        super().__init__(
            mobject,
            shift=point - mobject.get_center(),
            scale=0,
            **kwargs,
        )


class FadeTransform(Transform):
    def __init__(
        self,
        mobject: Mobject,
        target_mobject: Mobject,
        stretch: bool = True,
        dim_to_match: int = 1,
        **kwargs
    ):
        self.to_add_on_completion = target_mobject
        self.stretch = stretch
        self.dim_to_match = dim_to_match

        mobject.save_state()
        super().__init__(Group(mobject, target_mobject.copy()), **kwargs)

    def begin(self) -> None:
        self.ending_mobject = self.mobject.copy()
        Animation.begin(self)
        # Both 'start' and 'end' consists of the source and target mobjects.
        # At the start, the traget should be faded replacing the source,
        # and at the end it should be the other way around.
        start, end = self.starting_mobject, self.ending_mobject
        for m0, m1 in ((start[1], start[0]), (end[0], end[1])):
            self.ghost_to(m0, m1)
        # The two ends only became what they are here, so what a blend between them comes to
        # has to be settled again, having been settled over what they held a moment ago
        self.prepare_interpolation()

    def ghost_to(self, source: Mobject, target: Mobject) -> None:
        source.replace(target, stretch=self.stretch, dim_to_match=self.dim_to_match)
        source.set_uniform(**target.get_uniforms())
        source.set_opacity(0)

    def get_all_mobjects(self) -> list[Mobject]:
        return [
            self.mobject,
            self.starting_mobject,
            self.ending_mobject,
        ]

    def get_all_families_zipped(self) -> zip[tuple[Mobject]]:
        return Animation.get_all_families_zipped(self)

    def get_interpolation_ends(self) -> tuple[Mobject, Mobject]:
        # Its own pair rather than Transform's, this one never having made a target_copy
        return self.starting_mobject, self.ending_mobject

    def clean_up_from_scene(self, scene: Scene) -> None:
        Animation.clean_up_from_scene(self, scene)
        scene.remove(self.mobject)
        self.mobject[0].restore()
        if not self.remover:
            scene.add(self.to_add_on_completion)


class FadeTransformPieces(FadeTransform):
    def begin(self) -> None:
        self.mobject[0].align_family(self.mobject[1])
        super().begin()

    def ghost_to(self, source: Mobject, target: Mobject) -> None:
        for sm0, sm1 in zip(source.get_family(), target.get_family()):
            super().ghost_to(sm0, sm1)


class VFadeIn(Animation):
    """
    VFadeIn and VFadeOut only work for VMobjects,
    """
    def __init__(self, vmobject: VMobject, suspend_mobject_updating: bool = False, **kwargs):
        super().__init__(
            vmobject,
            suspend_mobject_updating=suspend_mobject_updating,
            **kwargs
        )

    def interpolate_submobject(
        self,
        submob: VMobject,
        start: VMobject,
        alpha: float
    ) -> None:
        submob.set_stroke(
            opacity=interpolate(0, start.get_stroke_opacity(), alpha)
        )
        submob.set_fill(
            opacity=interpolate(0, start.get_fill_opacity(), alpha)
        )


class VFadeOut(VFadeIn):
    def __init__(
        self,
        vmobject: VMobject,
        remover: bool = True,
        final_alpha_value: float = 0.0,
        **kwargs
    ):
        super().__init__(
            vmobject,
            remover=remover,
            final_alpha_value=final_alpha_value,
            **kwargs
        )

    def interpolate_submobject(
        self,
        submob: VMobject,
        start: VMobject,
        alpha: float
    ) -> None:
        super().interpolate_submobject(submob, start, 1 - alpha)


class VFadeInThenOut(VFadeIn):
    def __init__(
        self,
        vmobject: VMobject,
        rate_func: Callable[[float], float] = there_and_back,
        remover: bool = True,
        final_alpha_value: float = 0.5,
        **kwargs
    ):
        super().__init__(
            vmobject,
            rate_func=rate_func,
            remover=remover,
            final_alpha_value=final_alpha_value,
            **kwargs
        )
