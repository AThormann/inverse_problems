import numpy as np
import matplotlib.pyplot as plt


def pixel_intersection_length(x, y, receiverx, receivery, rayvectorx, rayvectory, debug=False):
    """
    Calculates the intersection lengths of a ray passing through a pixel grid.

    Parameters
    ----------
    x : np.ndarray
        1D array of x pixel centers.
    y : np.ndarray
        1D array of y pixel centers.
    receiverx : float
        The x position of the receiver.
    receivery : float
        The y position of the receiver.
    rayvectorx : float
        x component of vector pointing in the direction the ray is coming from.
    rayvectory : float
        y component of vector pointing in the direction the ray is coming from.
    debug : bool, optional
        If True, generates a debug plot. Default is False.

    Returns
    -------
    rows,cols,lengths of intersecting pixels

    (C) Aslak Grinsted, Niels Bohr Institute 2025
    """

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # loop in pixel units:
    curx = (receiverx - x[0]) / dx
    cury = (receivery - y[0]) / dy  # top edge of first y  - should always be -0.5

    ux = rayvectorx / dx
    uy = rayvectory / dy

    dlr = np.sign(ux)
    dud = np.sign(uy)
    # FIRST: move to first hit of image extents
    if curx > len(x) - 0.5:
        if dlr >= 0:
            return np.array([]), np.array([]), np.array([])
        dt = (len(x) - 0.5 - curx) / ux
        curx = len(x) - 0.5
        cury += uy * dt
    if curx < -0.5:
        if dlr <= 0:
            return np.array([]), np.array([]), np.array([])
        dt = (-0.5 - curx) / ux
        curx = -0.5
        cury += uy * dt
    if cury > len(y) - 0.5:
        if dud >= 0:
            return np.array([]), np.array([]), np.array([])
        dt = (len(y) - 0.5 - cury) / uy
        curx += ux * dt
        cury = len(y) - 0.5
    if cury < -0.5:
        if dud <= 0:
            return np.array([]), np.array([]), np.array([])
        dt = (-0.5 - cury) / uy
        curx += ux * dt
        cury = -0.5

    hitlist = []

    tol = max(len(x), len(y)) * np.finfo(float).eps
    while (
        (curx + dlr * tol >= -0.5)
        & (curx + dlr * tol <= len(x) - 0.5)
        & (cury + dud * tol >= -0.5)
        & (cury + dud * tol <= len(y) - 0.5)
        & (len(hitlist) < 100000)
    ):  # TODO: remove this check! - just there to stop infinite loops
        bottomy = dud * (np.floor(dud * cury + 0.5) + 0.5)  # next pixel bottom edge
        if ux != 0:
            sidex = dlr * (np.floor(dlr * curx + 0.5) + 0.5)
            Dx = sidex - curx
            dt = Dx / ux
            Dy = uy * dt
            if np.abs(Dy) < np.abs(bottomy - cury):
                hitlist.append([curx + Dx / 2, cury + Dy / 2, Dx, Dy])
                cury += Dy
                curx = sidex
                continue
        # now do pixel bottom hit
        Dy = bottomy - cury
        dt = Dy / uy
        Dx = ux * dt
        hitlist.append([curx + Dx / 2, cury + Dy / 2, Dx, Dy])
        cury = bottomy
        curx += Dx
    hitlist = np.array(hitlist)

    if len(hitlist) == 0:
        return np.array([]), np.array([]), np.array([])  # no intersections

    c = (hitlist[:, 0] + 0.5).astype(int)
    r = (hitlist[:, 1] + 0.5).astype(int)

    # convert back to real units
    hitlist = hitlist * np.array([dx, dy, dx, dy]) + np.array([x[0], y[0], 0, 0])

    L = np.sqrt(hitlist[:, 2] ** 2 + hitlist[:, 3] ** 2)

    if debug:
        Z = np.zeros((len(y), len(x)))
        Z[r, c] = L
        X, Y = np.meshgrid(x, y)
        plt.pcolormesh(X, Y, Z, edgecolor="k", lw=0.2, cmap="Reds")
        plt.plot(hitlist[:, 0], hitlist[:, 1], "rx")
        plt.gca().invert_yaxis()
        plt.plot(receiverx, receivery, "r*", markersize=10)
        plt.axis("equal")
        plt.colorbar(label="Intersection length")
        plt.title("Ray Pixel Intersection debug plot")

    return r, c, L
