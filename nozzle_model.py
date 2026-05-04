"""
nozzle_model.py
Quasi-1D isentropic flow solver for a de Laval (converging-diverging) rocket nozzle.
This simulator takes in a nozzle shape (via Bezier control points), computes the flow
field (i.e. Mach, pressure), computes performance (thrust coeffiicent C_F with a divergence
-loss correction based on the exit wall angle), and returns the results/plots things. 
note to selves for the paper: we used the following reference for guidance:
- Sutton & Biblarz, "Rocket Propulsion Elements" (thrust coefficient derivation)
"""

import numpy as np
from scipy.optimize import brentq
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def bezier_curve(control_points, n_pts=200):
    """
    The nozzle contour is defined using a Bezier curve where the inputs are the control point radii
    (these are the design variables) and the output is a smooth radius called r(x) along the nozzle.
    This is meant to turn the optimization into a low-dimensional design problem/should make it easier
    to solve.

    @param: control_points (these are the design variables): array-like, shape (n+1, 2)
    Control points [[x0, r0], [x1, r1], ...].
    @param: n_pts (int): Number of evaluation points along the curve (hardcoded at 200 for right now).

    @return: curve: ndarray, shape (n_pts, 2) Evaluated [x, r] coordinates along the curve.
    """
    cp = np.asarray(control_points, dtype=float)
    n = len(cp) - 1                           #polynomial degree
    t = np.linspace(0, 1, n_pts).reshape(-1, 1)
    curve = np.zeros((n_pts, 2))
    for i in range(n + 1):
        B_i = comb(n, i) * (1 - t) ** (n - i) * t ** i
        curve += B_i * cp[i]
    return curve


def bezier_derivative(control_points, n_pts=200):
    r"""
    First derivative of a Bézier curve. This computes the slope of the nozzle wall
    dr/dx along the contour. For our purposes, it's used to get the exit wall angle
    (this determines the divergence loss factor \lambda in the thrust calculation) 

    The derivative of a degree-n Bézier is a degree-(n-1) Bézier with
    control points  Q_i = n · (P_{i+1} − P_i).
    """
    cp = np.asarray(control_points, dtype=float)
    n = len(cp) - 1
    deriv_cp = n * np.diff(cp, axis=0)
    return bezier_curve(deriv_cp, n_pts)


#Isentropic Flow Relations are now defined as follows:

def area_mach_relation(M, gamma):
    """Return A/A* for a given Mach number (isentropic, quasi-1D). This maps flow properties → geometry.
    In our solver, we INVERT this relation to go from geometry (area ratio) → Mach number.
    
    @param: M (float): Mach number
    @param: gamma (float): ratio of specific heats

    @return: area ratio A/A*
    """
    g = gamma
    exp = (g + 1) / (2 * (g - 1))
    return (1.0 / M) * ((2.0 / (g + 1)) * (1.0 + (g - 1) / 2.0 * M ** 2)) ** exp


def solve_mach(area_ratio, gamma, supersonic=True):
    """
    Solves the area–Mach relation numerically (using area_mach_relation)
    numerically to find the Mach number. Since the relation cannot be inverted
    analytically, we use root finding (brentq) to solve area A/A* = f(M) (remember
    that we're going from area ratio --> Mach number). 
    There are two possible solutions: subsonic (M < 1) and supersonic (M > 1).

    @param: area_ratio (float): A / A* (must be >= 1)
    @param: gamma (float): ratio of specific heats
    @param: supersonic (bool): whether to return supersonic or subsonic root

    @return: M (float): Mach number
    """
    if area_ratio < 1.0 - 1e-10:
        raise ValueError(f"Area ratio {area_ratio:.4f} < 1 is unphysical.")
    if np.isclose(area_ratio, 1.0, atol=1e-10):
        return 1.0
    f = lambda M: area_mach_relation(M, gamma) - area_ratio
    if supersonic:
        return brentq(f, 1.0 + 1e-12, 200.0)
    else:
        return brentq(f, 1e-6, 1.0 - 1e-12)


def pressure_ratio(M, gamma):
    """Computes the static-to-chamber pressure ratio p/p_c. This is used
    after solving for Mach to recover the pressure distribution along the 
    nozzle.
    @param: M (float): Mach number
    @param: gamma (float): ratio of specific heats

    @return: pressure ratio p/p_c
    """
    return (1.0 + (gamma - 1) / 2.0 * M ** 2) ** (-gamma / (gamma - 1))


def temperature_ratio(M, gamma):
    """Compute the static-to-total (chamber) temperature ratio  T/T_c.
    This is not currently used in the simulator class but included for 
    completeness if temperature-dependent quantities are needed later.

    @param: M (float): Mach number
    @param: gamma (float): ratio of specific heats

    @return: temperature ratio T/T_c
    """
    return (1.0 + (gamma - 1) / 2.0 * M ** 2) ** (-1)


#Thrust Coefficient is defined as follows:

def thrust_coefficient(M_e, gamma, epsilon, p_a_pc, lam=1.0):
    r"""
    Computes the nozzle thrust coefficient C_F. This combines:
    - momentum thrust (from exhaust velocity)
    - pressure thrust (from exit vs ambient pressure)
    - divergence loss correction (lambda)
    This is the main performance metric we are optimizing. 

    @param: M_e (float): exit Mach number
    @param: gamma (float): ratio of specific heats
    @param: epsilon (float): exit area ratio A_e / A_t
    @param: p_a_pc (float): ambient-to-chamber pressure ratio
    @param: lam (float): divergence correction factor

    @return: C_F (float)
    """
    g = gamma
    pe_pc = pressure_ratio(M_e, g)

    #Momentum contribution (ideal 1-D isentropic momentum contribution
    #to the thrust coefficient C_F)
    momentum = np.sqrt(
        2 * g ** 2 / (g - 1)
        * (2 / (g + 1)) ** ((g + 1) / (g - 1))
        * (1.0 - pe_pc ** ((g - 1) / g))
    ) #expression is: \frac{\dot{m}\, u_e}{p_c A_t}

    #Pressure contribution
    press = (pe_pc - p_a_pc) * epsilon

    return lam * momentum + press


#Nozzle Model (main class/simulator)

class NozzleModel:
    """
    Quasi-1D isentropic nozzle model with a Bézier-parameterized diverging contour.
    The high-level idea is that this class maps design variables (control point radii) → 
    nozzle geometry → flow solution → performance (C_F).
    The pipeline is:
    design vars → Bézier curve → radius r(x) → area ratio A/A* → Mach distribution →
    pressure distribution → exit conditions → thrust coefficient C_F

    Geometry:
    - The throat is fixed at x = 0 with radius r_throat
    - The exit is fixed at x = L_nozzle
    - The shape between them is defined by a Bézier curve

    Design variables:
    - Radial coordinates of all control points except the throat
    - These are what the optimizer modifies

    Physics assumptions:
    - Quasi-1D, steady, isentropic flow
    - Always solves for the supersonic branch (after the throat)
    - No shocks or viscous effects are modeled

    Performance metric:
    - Thrust coefficient C_F (dimensionless)
    - Includes:
        - momentum thrust (ideal 1-D)
        - pressure thrust
        - divergence loss correction based on exit angle

    Notes:
    This is a black-box function from design vars --> C_F, and it's intended
    for use in derivative-free optimization (e.g. Bayesian optimization)
    """

    def __init__(
        self,
        r_throat=0.05,
        L_nozzle=0.20,
        gamma=1.3,
        p_a_over_p_c=0.0,
        n_control_points=5,
        n_stations=200,
    ):
        self.r_throat = r_throat
        self.L_nozzle = L_nozzle
        self.gamma = gamma
        self.p_a_over_p_c = p_a_over_p_c
        self.n_cp = n_control_points
        self.n_stations = n_stations

        #Fixed axial positions of control points (evenly spaced)
        self.x_cp = np.linspace(0.0, L_nozzle, n_control_points)

        #Number of free design variables (all radii EXCEPT the throat)
        self.n_vars = n_control_points - 1

        #Physically reasonable bounds on each radius
        self.r_min = r_throat * 1.01
        self.r_max = r_throat * 6.0
        self.bounds = [(self.r_min, self.r_max)] * self.n_vars

    #helpers

    def _build_control_points(self, design_vars):
        """Combine fixed throat with free design-variable radii."""
        cp = np.column_stack([
            self.x_cp,
            np.concatenate([[self.r_throat], design_vars]),
        ])
        return cp

    #full evaluation

    def evaluate(self, design_vars):
        """
        Runs the full physics pipeline for a given nozzle design.
        Steps:
        1. Build nozzle geometry (using the Bezier curve)
        2. Convert radius → area ratio
        3. Solve for Mach number at each axial station
        4. Compute pressure distribution
        5. Compute exit conditions and thrust coefficient
        **This is the core function used by the optimizer.
        @param: design_vars (array-like): control point radii
        @return: dict containing performance, flow field, and geometry
        """
        design_vars = np.asarray(design_vars, dtype=float)
        cp = self._build_control_points(design_vars)

        #contour & derivative
        contour = bezier_curve(cp, self.n_stations)
        deriv = bezier_derivative(cp, self.n_stations)
        x, r = contour[:, 0], contour[:, 1]

        #Validity: radius must stay ≥ throat everywhere
        if np.any(r < self.r_throat * 0.999):
            return self._invalid()

        #area ratios
        AR = (r / self.r_throat) ** 2
        AR = np.clip(AR, 1.0, None)          #numerical safety at throat

        #solve Mach at every station
        M = np.empty_like(AR)
        try:
            for i in range(len(AR)):
                M[i] = solve_mach(AR[i], self.gamma, supersonic=True)
        except Exception:
            return self._invalid()

        #exit conditions
        M_exit = M[-1]
        eps = AR[-1]
        pe_pc = pressure_ratio(M_exit, self.gamma)

        #divergence loss
        dx_dt, dr_dt = deriv[-1, 0], deriv[-1, 1]
        theta_exit = np.arctan2(abs(dr_dt), abs(dx_dt))   #output unit in rad
        lam = (1.0 + np.cos(theta_exit)) / 2.0

        #thrust coefficient
        CF = thrust_coefficient(M_exit, self.gamma, eps,
                                self.p_a_over_p_c, lam)

        #pressure distribution
        p_dist = np.array([pressure_ratio(m, self.gamma) for m in M])

        return {
            "C_F":           CF,
            "M_exit":        M_exit,
            "epsilon":       eps,
            "p_e_over_p_c":  pe_pc,
            "theta_exit":    np.degrees(theta_exit),
            "lambda_div":    lam,
            "contour":       contour,
            "mach":          M,
            "pressure":      p_dist,
            "valid":         True,
        }

    #objective for optimizer (minimize)

    def objective(self, design_vars):
        """Return −C_F  (minimize to maximize thrust coefficient)."""
        res = self.evaluate(design_vars)
        if not res["valid"]:
            return 1e6            #heavy penalty
        return -res["C_F"]

    #plotting

    def plot(self, design_vars, filename="nozzle_design.png", title=None):
        """
        Generates plots for a given nozzle design.
        Plots nozzle contour, mach number distribution, and pressure
        distribution all in the same window:
        Used for debugging and visualization (not part of optimization).
        """
        res = self.evaluate(design_vars)
        if not res["valid"]:
            print("Invalid design – cannot plot.")
            return

        x = res["contour"][:, 0]
        r = res["contour"][:, 1]
        cp = self._build_control_points(design_vars)

        if title is None:
            title = f"Nozzle Design  |  $C_F$ = {res['C_F']:.5f}"

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        fig.suptitle(title, fontsize=14, y=0.98)

        #(a) Contour
        ax = axes[0]
        ax.fill_between(x, -r, r, alpha=0.12, color="steelblue")
        ax.plot(x, r, "steelblue", lw=2, label="Wall")
        ax.plot(x, -r, "steelblue", lw=2)
        ax.plot(cp[:, 0], cp[:, 1], "ro--", ms=6, label="Control pts")
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel("Radius  [m]")
        ax.legend(loc="upper left")
        ax.set_title("Nozzle Contour")
        ax.set_aspect("equal")

        #(b) Mach
        ax = axes[1]
        ax.plot(x, res["mach"], color="darkorange", lw=2)
        ax.set_ylabel("Mach number")
        ax.set_title("Mach-Number Distribution")
        ax.grid(True, alpha=0.25)

        #(c) Pressure
        ax = axes[2]
        ax.plot(x, res["pressure"], color="seagreen", lw=2)
        ax.set_ylabel("$p / p_c$")
        ax.set_xlabel("Axial position  [m]")
        ax.set_title("Static / Chamber Pressure Ratio")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        #plt.show() showing causing issues because of FigureCanvasAgg
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def print_summary(self, design_vars):
        """
        Prints a summary of the nozzle performance.
        Includes thrust coefficient, exit Mach number, area ratio,
        pressure ratio, and divergence loss factor.
        """
        res = self.evaluate(design_vars)
        if not res["valid"]:
            print("Invalid design.")
            return
        print(f"Thrust coefficient C_F = {res['C_F']:.6f}")
        print(f"Exit Mach number = {res['M_exit']:.4f}")
        print(f"Exit area ratio ε = {res['epsilon']:.4f}")
        print(f"Exit pressure ratio p_e/p_c= {res['p_e_over_p_c']:.6f}")
        print(f"Exit wall angle = {res['theta_exit']:.2f}°")
        print(f"Divergence factor \lambda = {res['lambda_div']:.6f}")

    #internal (cannot be used outside of this class)

    @staticmethod
    def _invalid():
        """
        Returns a standardized invalid result.
        Used when the geometry is unphysical or the solver fails.
        @return: dict with invalid flag
        """
        return {"C_F": -999.0, "valid": False}


#Quick validation test

if __name__ == "__main__":
    '''NOTE: this is just a test to see if the simulator works. This is NOT our baseline or anything
    we will actually use for the results (see baseline_nozzle.py for that)'''
    model = NozzleModel(
        #we can tweak these params if wanted!!
        r_throat=0.05,          #5 cm throat radius
        L_nozzle=0.20,          #20 cm diverging section
        gamma=1.3,              #typical rocket exhaust
        p_a_over_p_c=0.01,      #near-vacuum (high altitude)
        n_control_points=5,     #→ 4 design variables
    )

    print(f"Design variables: {model.n_vars}")
    print(f"Bounds per var: {model.bounds[0]}")

    #Test: gentle linear expansion to 2.5× throat radius
    r_exit = model.r_throat * 2.5
    dv = np.linspace(model.r_throat * 1.2, r_exit, model.n_vars)
    print(f"Test radii: {np.round(dv, 4)}")

    model.print_summary(dv)
    model.plot(dv, filename="nozzle_test.png", title="Test – Linear Expansion")
    print("Plot saved as nozzle_test.png")