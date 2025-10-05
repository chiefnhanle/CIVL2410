import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection,LineCollection
def solve(k, bc, Lx, max_iter=10000, tol=1e-4):
    """
    Solve ∇·(K ∇h) = 0 for steady-state seepage using finite differences.

    Parameters:
        domain: dict with fields:
            - 'h': initial head field (ny, nx)
            - 'k': permeability field (ny, nx)
            - 'bc': boundary conditions (dict)
        max_iter: maximum number of iterations
        tol: convergence criterion (max |Δh|)

    Returns:
        h: updated head field (ny, nx)
        vx, vy: Darcy velocity components (ny, nx)
    """

    ny, nx = k.shape

    # Set up the initial head field
    h = np.zeros((ny, nx))
    h = apply_boundary_conditions(h, bc)

    converged = False

    for iteration in range(max_iter):
        max_change = 0.0

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                old_val = h[j, i]

                # Harmonic means for K at interfaces
                k_e = harmonic(k[j, i], k[j, i + 1])
                k_w = harmonic(k[j, i], k[j, i - 1])
                k_n = harmonic(k[j, i], k[j - 1, i])
                k_s = harmonic(k[j, i], k[j + 1, i])

                w_sum = k_e + k_w + k_n + k_s
                if w_sum == 0:
                    continue  # fully isolated point

                h[j, i] = (
                    k_e * h[j, i + 1]
                    + k_w * h[j, i - 1]
                    + k_n * h[j - 1, i]
                    + k_s * h[j + 1, i]
                ) / w_sum

                max_change = max(max_change, abs(h[j, i] - old_val))

        h = apply_boundary_conditions(h, bc)

        if max_change < tol:
            converged = True
            print(f"Converged in {iteration + 1} iterations.")
            break

    if not converged:
        print(f"Warning: did not converge within {max_iter} iterations.")

    dx = Lx / (nx - 1)
    dy = Lx / (ny - 1)
    vx, vy = get_velocity(k, h, dx, dy)
    return h, vx, vy


def apply_boundary_conditions(h, bc, fixed_points=None, fixed_regions=None):
    """
    Apply boundary conditions and internal constraints to the hydraulic head field.

    Parameters:
        h (numpy.ndarray): 2D array for hydraulic head distribution.
        bc (dict): Boundary conditions {"left","right","top","bottom"}.
        Dirichlet if value is not None, Neumann if None.
        fixed_points (list of tuples): [(i,j,value), ...] for individual cells.
        fixed_regions (list of tuples): [(slice_y, slice_x, value), ...] for rectangular regions.

    Returns:
        numpy.ndarray: Updated head field.
    """
    ny, nx = h.shape

    # --- Boundaries ---
    if bc.get("left") is not None:
        h[:, 0] = bc["left"]
    else:
        h[:, 0] = h[:, 1]

    if bc.get("right") is not None:
        h[:, -1] = bc["right"]
    else:
        h[:, -1] = h[:, -2]

    if bc.get("top") is not None:
        h[0, :] = bc["top"]
    else:
        h[0, :] = h[1, :]

    if bc.get("bottom") is not None:
        h[-1, :] = bc["bottom"]
    else:
        h[-1, :] = h[-2, :]

    # --- Internal fixed points ---
    if fixed_points:
        for i, j, val in fixed_points:
            h[i, j] = val

    # --- Internal fixed regions ---
    if fixed_regions:
        for sy, sx, val in fixed_regions:
            h[sy, sx] = val

    return h


def get_velocity(k, h, dx, dy):
    """
    Calculate the Darcy velocity components from the hydraulic head field.
    The velocity components are calculated using the finite difference method.
    The velocity is given by:
        vx = -K * (∂h/∂x)
        vy = K * (∂h/∂y)
    where K is the hydraulic conductivity and h is the hydraulic head.

    Parameters:
        k (numpy.ndarray): 2D array of hydraulic conductivity values.
        h (numpy.ndarray): 2D array representing the hydraulic head distribution.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
    Returns:
        vx (numpy.ndarray): 2D array of x-component of velocity.
        vy (numpy.ndarray): 2D array of y-component of velocity.
    """

    dh_dy, dh_dx = np.gradient(h, dy, dx)
    vx = -k * dh_dx
    vy = k * dh_dy

    return vx, vy
def add_k_outlines(ax, k, Lx, Ly, lw_min=0, lw_max=3.0, color='g', alpha=0.3):
    k = np.asarray(k, dtype=float)
    ny, nx = k.shape
    dx, dy = Lx / nx, Ly / ny

    # transform
    k_log10 = -np.log10(k)

    # normalize to [0,1]
    vmin, vmax = k_log10.min(), k_log10.max()
    norm = (k_log10 - vmin) / (vmax - vmin)

    # map to linewidth
    k_lw = lw_min + norm * (lw_max - lw_min)

    # draw outlines
    for j in range(ny):
        for i in range(nx):
            rect = Rectangle((i*dx, j*dy), dx, dy,
            linewidth=k_lw[j, i],
            edgecolor=color,
            facecolor='none',
            alpha=alpha)
            ax.add_patch(rect)

def flownet_upgraded(h, k, bc, Lx, Nf=5, Nh=5, filename=None,lw_min=0.00, lw_max=10, title = "Seepage flow"):
    """
    Plots the results of a seepage flow simulation, including the hydraulic head
    distribution and flow streamlines.

    Parameters:
        h (numpy.ndarray): 2D array representing the hydraulic head distribution.
        k (numpy.ndarray): 2D array of hydraulic conductivity values.
        bc (dict): Dictionary containing boundary conditions with keys "left", "right", "top", "bottom".
        Lx (float): Length of the domain in the x-direction.
        Nf (int): Number of flow lines to plot.
        Nh (int): Number of head contours to plot.
        filename (str): Optional filename to save the plot. If None, the plot will be displayed.

    Returns:
        None
    """
    ny, nx = h.shape
    Ly = ny / nx * Lx
    x, y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    dpi = 100
    target_width = 8  # inches
    target_height = target_width * (ny / nx)
    plt.figure(figsize=(target_width, target_height), dpi=dpi, layout="constrained")

    # Determine seed points from Dirichlet boundaries (i.e. where water enters). Aim to have Nf points in total, but no guarantee that they are in the right places.
    seed_points = []

    # iterate over the keys in the bc dictionary and check if they are not None
    sides_with_inflow = 0
    for key in bc:
        if (
            bc[key] is not None and bc[key] != 0
        ):  # assuming non-zero values are inflow --- this is a guess
            sides_with_inflow += 1

    seeds_per_side = (Nf - 1) // sides_with_inflow if sides_with_inflow > 0 else 0

    if bc.get("left") is not None:
        ys = np.linspace(2 * dy, Ly - 2 * dy, seeds_per_side)
        seed_points.extend(np.column_stack([np.full(seeds_per_side, 2 * dx), ys]))
    if bc.get("right") is not None:
        ys = np.linspace(2 * dy, Ly - 2 * dy, seeds_per_side)
        seed_points.extend(np.column_stack([np.full(seeds_per_side, Lx - 2 * dx), ys]))
    if bc.get("top") is not None:
        xs = np.linspace(2 * dx, Lx - 2 * dx, seeds_per_side)
        seed_points.extend(np.column_stack([xs, np.full(seeds_per_side, 2 * dy)]))
    if bc.get("bottom") is not None:
        xs = np.linspace(2 * dx, Lx - 2 * dx, seeds_per_side)
        seed_points.extend(np.column_stack([xs, np.full(seeds_per_side, Ly - 2 * dy)]))

    seed_points = np.array(seed_points)

    h[k == 0] = np.nan  # set zero k to NaN so that it doesn't show up in the plot
    
    vx, vy = get_velocity(k, h, dx, dy)

    plt.contourf(x, y, h, levels=Nh, cmap="coolwarm")
    plt.colorbar(label="Head (m)")
    # >>> NEW: outlines scaled by k (low k => thicker) <<<
    ax = plt.gca()
# draw outlines (thickest at k=1e-10, thinnest at k=1e1 if those are in your data)
    add_k_outlines(ax, k, Lx=Lx, Ly=Ly, lw_min=lw_min, lw_max=lw_max)
    if seed_points.size > 0:
        plt.streamplot(x, y, vx, -vy, color="k", start_points=seed_points, linewidth=1)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().invert_yaxis()
    print("i'm upgraded")
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def harmonic(a, b):
    """
    Calculate the harmonic mean of two numbers.

    The harmonic mean is defined as:
        H = 2 * a * b / (a + b)
    If the sum of `a` and `b` is zero, the function returns 0.0 to avoid division by zero.

    Parameters:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The harmonic mean of `a` and `b`, or 0.0 if `a + b` is zero.
    """
    return 2 * a * b / (a + b) if a + b > 0 else 0.0

