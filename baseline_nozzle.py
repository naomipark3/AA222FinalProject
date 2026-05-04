r"""
baseline_nozzle.py
Baseline 15° half-angle conical nozzle used as the reference benchmark for
the optimization study. This class wraps an existing NozzleModel instance and
hands it the specific design_vars vector that produces a straight-walled
(conical) contour when fed through the Bezier machinery in nozzle_model.py.
The baseline does NOT add any physics, it just produces one standard input
so the same evaluate() pipeline in the NozzleModel class gives us the "before" 
performance numbers that each optimized design will be compared against.
note to selves for the paper: 15° is the textbook conical reference half-angle
(Sutton & Biblarz, Chapter 3; see section 3.4, pg. 74 specifically). 
The divergence loss at 15° works out to \lambda = (1 + cos 15°)/2 ≈ 0.983, 
which our solver should reproduce automatically since it computes \lambda 
from the exit wall angle.
"""

import numpy as np
from nozzle_model import NozzleModel


#Baseline Nozzle (wrapper around NozzleModel)

class BaselineNozzle:
    """
    The baseline (conical) nozzle design is a straight diverging wall at a fixed
    half-angle. Throat radius, nozzle length, and control-point axial spacing
    are all *inherited directly from the NozzleModel instance passed in*, so the
    baseline lives in EXACTLY the same design space as the optimizer.
    The high-level idea is that a Bézier curve whose control points are
    collinear collapses to that straight line exactly (not approximately), so
    a conical nozzle can be expressed as one specific design_vars vector and
    pushed through the existing simulator with no code changes to the physics.

    Geometry:
    - Throat fixed at x = 0 with radius r_throat (from the model)
    - Exit at x = L_nozzle (from the model)
    - Wall radius along the diverging section: r(x) = r_throat + x · tan(α)

    Design variables:
    - The free control-point radii (all control points except the throat),
      placed on the straight conical line. Same shape/ordering as the
      design_vars the optimizer manipulates.

    Usage (i.e. how to call in main and how we will use it for the project):
    - Build a NozzleModel as usual.
    - Wrap it: baseline = BaselineNozzle(model)
    - Pull metrics: result = baseline.evaluate()
    - Compare results to optimized parameter or objective

    Notes:
    - This is NOT an optimizer or a separate solver, it produces ONE input.
    - half_angle_deg defaults to 15° (textbook standard). Adjustable if we
      ever want to sweep the baseline angle for sensitivity analysis.
    """

    def __init__(self, model, half_angle_deg=15.0):
        self.model = model
        self.half_angle_deg = half_angle_deg
        self.half_angle = np.radians(half_angle_deg)
        self.design_vars = self._build_design_vars()

    #helpers

    def _build_design_vars(self):
        """
        Build the design_vars vector for the conical baseline. Places each
        free control point on the straight line from the throat outward at
        slope tan(half_angle). Free control points = all NozzleModel control
        points except the throat (matching NozzleModel.n_vars exactly).
        @param: (none; uses self.model and self.half_angle)
        @return: design_vars: ndarray, shape (n_vars,) Free-variable radii in
        meters, ordered from throat-adjacent station to exit station.
        """
        #free axial stations (skip the throat at x = 0)
        x_free = self.model.x_cp[1:]

        #straight-line radius along the conical wall
        r_free = self.model.r_throat + x_free * np.tan(self.half_angle)

        #radii must stay within the model's own design bounds
        if np.any(r_free > self.model.r_max):
            raise ValueError(
                f"Baseline exit radius {r_free[-1]:.4f} m exceeds model r_max "
                f"({self.model.r_max:.4f} m). Try a smaller half-angle or a "
                f"longer nozzle."
            )
        if np.any(r_free < self.model.r_min):
            raise ValueError(
                f"Baseline radius {r_free.min():.4f} m falls below model r_min "
                f"({self.model.r_min:.4f} m). Half-angle is too small for this geometry."
            )

        return r_free

    #full evaluation

    def evaluate(self):
        """
        Run the baseline design through the existing NozzleModel pipeline.
        Returns the SAME dict shape that model.evaluate() returns for any
        other design — same keys, same units — so downstream code (objective
        extraction, plotting, comparison) is identical to the optimized case.

        @return: dict containing performance, flow field, and geometry
        (see NozzleModel.evaluate for the full key list).
        """
        return self.model.evaluate(self.design_vars)

    #plotting

    def plot(self, filename="baseline_nozzle.png"):
        """
        Generates the standard three-panel plot (contour + Mach + pressure)
        for the baseline design, with a baseline-specific title so it's
        easy to tell apart from the optimized-design plots in the writeup.
        """
        res = self.evaluate()
        title = (f"Baseline: {self.half_angle_deg:.1f}° Conical Nozzle; "
                 f"$C_F$ = {res['C_F']:.5f}")
        self.model.plot(self.design_vars, filename=filename, title=title)

    def print_summary(self):
        """
        Print baseline performance summary. Same fields as
        NozzleModel.print_summary with a header added so it's obvious
        when the results print out in the terminal which design is being reported.
        """
        print(f"Baseline ({self.half_angle_deg:.1f}° conical) relevant values:")
        print(f"Free-variable radii [m]: {np.round(self.design_vars, 4)}")
        self.model.print_summary(self.design_vars)


#Quick validation test

if __name__ == "__main__":

    #Same model config the optimizer will use — keep these in sync
    model = NozzleModel(
        r_throat=0.05,          #5 cm throat radius
        L_nozzle=0.20,          #20 cm diverging section
        gamma=1.3,              #typical rocket exhaust
        p_a_over_p_c=0.01,      #near-vacuum (high altitude)
        n_control_points=5,     #5 control points yields 4 design variables
    )

    baseline = BaselineNozzle(model, half_angle_deg=15.0)
    baseline.print_summary()
    baseline.plot(filename="baseline_test.png")
    print("Plot saved as baseline_test.png")