#!/usr/bin/env python
"""

PROCESS plot_proc using process_io_lib functions and MFILE.DAT

James Morris
13/04/2014
CCFE
Revised by Michael Kovari, 7/1/2016

24/11/2021: Global dictionary variables moved within the functions
to avoid cyclic dependencies. This is because the dicts
generation script imports, and inspects, process.

"""
import argparse
import os
from dataclasses import dataclass
from enum import Enum, auto
from importlib import resources
from operator import attrgetter
from typing import ClassVar, Dict, Literal, Optional, Tuple, Union
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.backends.backend_pdf as bpdf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.path import Path

import process.io.mfile as mf
from process.impurity_radiation import read_impurity_file
from process.io.python_fortran_dicts import get_dicts

if os.name == "posix" and "DISPLAY" not in os.environ:
    mpl.use("Agg")
mpl.rcParams["figure.max_open_warning"] = 40


solenoid = "pink"
cscompression = "red"
tfc = "cyan"
thermal_shield = "gray"
vessel = "green"
shield = "green"
blanket = "magenta"
plasma = "khaki"
cryostat = "red"
firstwall = "darkblue"
winding = "blue"
nbshield_colour = "gray"

thin = 0

RADIAL_BUILD = [
    "bore",
    "ohcth",
    "precomp",
    "gapoh",
    "tfcth",
    "tftsgap",
    "thshield_ib",
    "gapds",
    "d_vv_in",
    "shldith",
    "vvblgapi",
    "blnkith",
    "fwith",
    "scrapli",
    "rminori",
    "rminoro",
    "scraplo",
    "fwoth",
    "blnkoth",
    "vvblgapo",
    "shldoth",
    "d_vv_out",
    "gapsto",
    "thshield_ob",
    "tftsgap",
    "tfthko",
]

vertical_upper = [
    "rminor*kappa",
    "vgaptop",
    "fwtth",
    "blnktth",
    "vvblgap",
    "shldtth",
    "d_vv_top",
    "vgap2",
    "thshield_vb",
    "tftsgap",
    "tfcth",
]

vertical_lower = [
    "rminor*kappa",
    "vgap",
    "divfix",
    "shldlth",
    "d_vv_bot",
    "vgap2",
    "thshield_vb",
    "tftsgap",
    "tfcth",
]

ANIMATION_INFO = [
    ("rmajor", "Major radius", "m"),
    ("rminor", "Minor radius", "m"),
    ("aspect", "Aspect ratio", ""),
]

RTANGLE = np.pi / 2


@dataclass
class PlotData:
    bore: float
    ohcth: float
    gapoh: float
    tfcth: float
    gapds: float
    ddwi: float
    shldith: float
    blnkith: float
    fwith: float
    scrapli: float
    rmajor: float
    rminor: float
    scraplo: float
    fwoth: float
    blnkoth: float
    shldoth: float
    gapsto: float
    tfthko: float
    rdewex: float
    zdewex: float
    ddwex: float
    vvblgap: float
    d_vv_in: float
    d_vv_out: float
    fwtth: float

    # Magnets related
    n_tf: int
    wwp1: Optional[float]
    wwp2: Optional[float]
    dr_tf_wp: Optional[float]
    tinstf: Optional[float]
    thkcas: Optional[float]
    casthi: Optional[float]

    nbshield: float
    rtanbeam: float
    rtanmax: float
    beamwd: float

    # Pedestal profile parameters
    ipedestal: int
    neped: int
    nesep: int
    rhopedn: float
    rhopedt: float
    tbeta: float
    teped: float
    tesep: float
    alphan: float
    alphat: float
    ne0: float
    te0: float

    # Plasma
    triang: float
    triang95: float
    kappa95: float
    alphaj: float
    q0: float
    q95: float
    kallenbach_switch: int

    # rad profile
    ssync: float
    bt: float
    vol: float

    # power info
    pgrossmw: float
    pthermmw: float
    htpmw: float
    pnetelmw: float
    powfmw: float
    crypmw: float

    # flags
    i_single_null: int

    upper: Dict
    cumulative_upper: Dict

    lower: Dict
    cumulative_lower: Dict

    # Magnet conditionals
    mag_cond: ClassVar[tuple[str]] = (
        "wwp1",
        "wwp2",
        "dr_tf_wp",
        "tinstf",
        "thkcas",
        "casthi",
    )
    # Radial build keys
    rad_cond: ClassVar[tuple[str]] = (
        "upper",
        "lower",
        "cumulative_upper",
        "cumulative_lower",
    )

    @classmethod
    def from_mfile(cls, m_file: mf.MFile, scan):
        data = {}
        fields = cls.__dataclass_fields__.copy()
        for cond in (cls.mag_cond, cls.rad_cond):
            for v in cond:
                fields.pop(v, None)

        for var in fields:
            data[var] = m_file.data["bore"].get_scan(scan)
            if var == "n_tf":
                cls.process_tf(m_file, scan, data)
        (
            data["upper"],
            data["lower"],
            data["cumulative_upper"],
            data["cumulative_lower"],
        ) = cls.create_radial_build(m_file, scan)
        return cls(**data)

    @classmethod
    def process_tf(cls, m_file, scan, data):
        # Check for Copper magnets
        i_tf_sup = (
            int(m_file.data["i_tf_sup"].get_scan(scan))
            if "i_tf_sup" in m_file.data
            else 1
        )

        # Check integer turns
        i_tf_turns_integer = (
            int(m_file.data["i_tf_turns_integer"].get_scan(scan))
            if "i_tf_turns_integer" in m_file.data
            else 0
        )
        if i_tf_sup == 1:  # If superconducting magnets
            # casthi to be re-inergrated to resistives when in-plane stresses is integrated
            for k in cls.mag_cond:
                if k == "wwp2":
                    data[k] = (
                        m_file.data[k].get_scan(scan)
                        if i_tf_turns_integer == 0
                        else None
                    )
                data[k] = m_file.data[k].get_scan(scan)

        else:
            for k in cls.mag_cond:
                data[k] = None

    @staticmethod
    def create_radial_build(m_file, scan):
        # Build the dictionaries of radial and vertical build values and cumulative values

        # UNUSED
        # radial = {}
        # cumulative_radial = {}
        # subtotal = 0
        # for item in RADIAL_BUILD:
        #     if item == "rminori" or item == "rminoro":
        #         build = m_file.data["rminor"].get_scan(scan)
        #     elif item == "vvblgapi" or item == "vvblgapo":
        #         build = m_file.data["vvblgap"].get_scan(scan)
        #     elif "d_vv_in" in item:
        #         build = m_file.data["d_vv_in"].get_scan(scan)
        #     elif "d_vv_out" in item:
        #         build = m_file.data["d_vv_out"].get_scan(scan)
        #     else:
        #         build = m_file.data[item].get_scan(scan)

        # radial[item] = build
        # subtotal += build
        # cumulative_radial[item] = subtotal

        upper = {}
        cumulative_upper = {}
        subtotal = 0
        for item in vertical_upper:
            upper[item] = m_file.data[item].get_scan(scan)
            subtotal += upper[item]
            cumulative_upper[item] = subtotal

        lower = {}
        cumulative_lower = {}
        subtotal = 0
        for item in vertical_lower:
            lower[item] = m_file.data[item].get_scan(scan)
            subtotal -= lower[item]
            cumulative_lower[item] = subtotal

        return upper, lower, cumulative_upper, cumulative_lower

    def power_info(self):
        return {
            n: getattr(self, n)
            for n in (
                "pgrossmw",
                "pthermmw",
                "htpmw",
                "pnetelmw",
                "powfmw",
                "crypmw",
                "ipedestal",
            )
        }

    def info(self):
        pass


def plotdh(axis, r0, a, delta, kap):
    """Plots half a thin D-section, centred on z = 0.

    Arguments:
        axis --> axis object to plot to
        r0 --> major radius of centre
        a --> horizontal radius
        delta --> triangularity
        kap --> elongation

    Returns
    -------
        rs --> radial coordinates of D-section
        zs --> vertical coordinates of D-section
    """
    angs = np.linspace(0, np.pi, 50, endpoint=True)
    rs = r0 + a * np.cos(angs + delta * np.sin(1.0 * angs))
    zs = kap * a * np.sin(angs)
    axis.plot(rs, zs, color="black", lw=thin)
    return rs, zs


def plotdhgap(axis, inpt, outpt, inthk, outthk, toppt, topthk, delta, col):
    """Plots half a thick D-section with a gap.

    Arguments:
        axis --> axis object to plot to
        inpt --> inner points
        outpt --> outer points
        inthk --> inner thickness
        outthk --> outer thickness
        toppt --> top points
        topthk --> top thickness
        delta --> triangularity
        col --> color for fill

    """
    arc = np.pi / 4.0
    r01 = (inpt + outpt) / 2.0
    r02 = (inpt + inthk + outpt - outthk) / 2.0
    a1 = r01 - inpt
    a2 = r02 - inpt - inthk
    kap1 = toppt / a1
    kap2 = (toppt - topthk) / a2
    # angs = ((np.pi/2.) - arc/2.) * findgen(50)/49.
    angs = np.linspace(0.0, (np.pi / 2.0) - arc / 2.0, 50, endpoint=True)
    rs1 = r01 + a1 * np.cos(angs + delta * np.sin(angs))
    zs1 = kap1 * a1 * np.sin(angs)
    rs2 = r02 + a2 * np.cos(angs + delta * np.sin(angs))
    zs2 = kap2 * a2 * np.sin(angs)
    # angs = !pi + ((!pi/2.) - arc) * findgen(50)/49.
    angs = np.linspace(np.pi, np.pi + ((np.pi / 2.0) - arc), 50, endpoint=True)
    rs3 = r01 + a1 * np.cos(angs + delta * np.sin(angs))
    zs3 = kap1 * a1 * np.sin(angs)
    rs4 = r02 + a2 * np.cos(angs + delta * np.sin(angs))
    zs4 = kap2 * a2 * np.sin(angs)

    axis.plot(
        np.concatenate([rs1, rs2[::-1]]),
        np.concatenate([zs1, zs2[::-1]]),
        color="black",
        lw=thin,
    )
    axis.plot(
        np.concatenate([rs3, rs4[::-1]]),
        -np.concatenate([zs3, zs4[::-1]]),
        color="black",
        lw=thin,
    )
    axis.fill(
        np.concatenate([rs1, rs2[::-1]]), np.concatenate([zs1, zs2[::-1]]), color=col
    )
    axis.fill(
        np.concatenate([rs3, rs4[::-1]]), -np.concatenate([zs3, zs4[::-1]]), color=col
    )


def plot_plasma(axis, plot_data):
    """Plots the plasma boundary arcs.

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE data object
        scan --> scan number to use

    """
    r0 = plot_data.rmajor
    a = plot_data.rminor
    delta = 1.5 * plot_data.triang95
    kappa = (1.1 * plot_data.kappa95) + 0.04
    i_single_null = plot_data.i_single_null

    x1 = (2.0 * r0 * (1.0 + delta) - a * (delta**2 + kappa**2 - 1.0)) / (
        2.0 * (1.0 + delta)
    )
    x2 = (2.0 * r0 * (delta - 1.0) - a * (delta**2 + kappa**2 - 1.0)) / (
        2.0 * (delta - 1.0)
    )
    r1 = 0.5 * np.sqrt(
        (a**2 * ((delta + 1.0) ** 2 + kappa**2) ** 2) / ((delta + 1.0) ** 2)
    )
    r2 = 0.5 * np.sqrt(
        (a**2 * ((delta - 1.0) ** 2 + kappa**2) ** 2) / ((delta - 1.0) ** 2)
    )
    theta1 = np.arcsin((kappa * a) / r1)
    theta2 = np.arcsin((kappa * a) / r2)
    inang = 1.0 / r1
    outang = 1.5 / r2
    if i_single_null == 0:
        angs1 = np.linspace(
            -(inang + theta1) + np.pi, (inang + theta1) + np.pi, 256, endpoint=True
        )
        angs2 = np.linspace(-(outang + theta2), (outang + theta2), 256, endpoint=True)
    elif i_single_null < 0:
        angs1 = np.linspace(
            -(inang + theta1) + np.pi, theta1 + np.pi, 256, endpoint=True
        )
        angs2 = np.linspace(-theta2, (outang + theta2), 256, endpoint=True)
    else:
        angs1 = np.linspace(
            -theta1 + np.pi, (inang + theta1) + np.pi, 256, endpoint=True
        )
        angs2 = np.linspace(-(outang + theta2), theta2, 256, endpoint=True)

    xs1 = -(r1 * np.cos(angs1) - x1)
    ys1 = r1 * np.sin(angs1)
    xs2 = -(r2 * np.cos(angs2) - x2)
    ys2 = r2 * np.sin(angs2)
    axis.plot(xs1, ys1, color="black")
    axis.plot(xs2, ys2, color="black")
    axis.fill_betweenx(
        ys1,
        xs1,
        xs2,
        where=(xs2 < xs1) & (ys1 > (-a * kappa)) & (ys1 < (a * kappa)),
        color=plasma,
    )
    axis.fill_betweenx(ys1, xs1, xs2, where=(xs2 > xs1), color="none")


def plot_centre_cross(axis, rmajor):
    """Function to plot centre cross on plot

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE data object
        scan --> scan number to use
    """
    axis.plot(
        [rmajor - 0.25, rmajor + 0.25, rmajor, rmajor, rmajor],
        [0, 0, 0, 0.25, -0.25],
        color="black",
    )


def cumulative_radial_build(
    section, mfile_data, plot_data, scan, *, with_previous=False, verbose=True
) -> Union[float, Tuple[float, float]]:
    """Function for calculating the cumulative radial build up to and
    including the given section.

    Arguments:
        section --> section of the radial build to go up to
        mfile_data --> MFILE data object
        scan --> scan number to use

    Returns
    -------
        cumulative_build --> cumulative radial build up to and including
                             section given
        previous         --> cumulative radial build up to section given

    """
    complete = False
    cumulative_build = 0
    build = 0
    for item in RADIAL_BUILD:
        if item == "rminori" or item == "rminoro":
            build = plot_data.rminor
        elif item == "vvblgapi" or item == "vvblgapo":
            build = plot_data.vvblgap
        elif "d_vv_in" in item:
            build = plot_data.d_vv_in
        elif "d_vv_out" in item:
            build = plot_data.d_vv_out
        else:
            build = mfile_data.data[item].get_scan(scan)
        cumulative_build += build
        if item == section:
            complete = True
            break

    if not complete and verbose:
        print("radial build parameter ", section, " not found")
    if with_previous:
        return cumulative_build, cumulative_build - build
    else:
        return cumulative_build


def poloidal_cross_section(axis, mfile_data, plot_data, scan, demo_ranges):
    """Function to plot poloidal cross-section

    Arguments:
      axis --> axis object to add plot to
      mfile_data --> MFILE data object
      scan --> scan number to use

    """
    axis.set_xlabel("R / m")
    axis.set_ylabel("Z / m")
    axis.set_title("Poloidal cross-section")

    plot_vacuum_vessel(axis, mfile_data, plot_data, scan)
    plot_shield(axis, mfile_data, plot_data, scan)
    plot_blanket(axis, mfile_data, plot_data, scan)
    plot_firstwall(axis, mfile_data, plot_data, scan)

    plot_plasma(axis, plot_data)
    plot_centre_cross(axis, plot_data.rmajor)
    plot_cryostat(axis, mfile_data, plot_data, scan)

    plot_tf_coils(axis, mfile_data, plot_data, scan)
    plot_pf_coils(axis, mfile_data, scan)

    # Ranges
    if demo_ranges:
        # DEMO : Fixed ranges for comparison
        axis.set_ylim([-15, 15])
        axis.set_xlim([0, 20])
    else:
        # Adapatative ranges
        axis.set_xlim([0, axis.get_xlim()[1]])


def plot_cryostat(axis, mfile_data, plot_data, scan):
    """Function to plot cryostat in poloidal cross-section"""
    rdewex = plot_data.rdewex
    ddwex = plot_data.ddwex
    zdewex = plot_data.zdewex

    for i in (1, -1):
        axis.add_patch(
            patches.Rectangle(
                [rdewex, 0], ddwex, i * (zdewex + ddwex), lw=0, facecolor=cryostat
            )
        )
        axis.add_patch(
            patches.Rectangle(
                [0, i * zdewex], rdewex, i * ddwex, lw=0, facecolor=cryostat
            )
        )


def color_key(axis):
    """Function to plot the colour key
    Arguments:
      axis --> object to add plot to
    """
    axis.set_ylim([0, 10])
    axis.set_xlim([0, 10])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)
    data = {
        "CS coil": (10, 9.7, solenoid),
        "CS comp": (9, 8.7, cscompression),
        "TF coil": (8, 7.7, tfc),
        "Th shield": (7, 6.7, thermal_shield),
        "VV & shield": (6, 5.7, vessel),
        "Blanket": (5, 4.7, blanket),
        "First wall": (4, 3.7, firstwall),
        "Plasma": (3, 2.7, plasma),
        "PF coils": (2, 1.7, "none"),
        "NB duct shield": (1, 0.7, nbshield_colour),
        "cryostat": (0.1, -0.3, cryostat),
    }
    for label, (z, rect_h, colour) in data.items():
        axis.text(-5, z, label, ha="left", va="top", size="medium")
        if colour == "none":
            axis.add_patch(
                patches.Rectangle(
                    [0.2, rect_h], 1, 0.4, lw=1, facecolor=colour, edgecolor="black"
                )
            )

        else:
            axis.add_patch(
                patches.Rectangle([0.2, rect_h], 1, 0.4, lw=0, facecolor=colour)
            )


def toroidal_cross_section(axis, mfile_data, plot_data, colour_dict, scan, demo_ranges):
    """Function to plot toroidal cross-section
    Arguments:
      axis --> axis object to add plot to
      mfile_data --> MFILE data object
      scan --> scan number to use
    """
    attrs = (
        "rmajor",
        "rminor",
        "rdewex",
        "ddwex",
        "n_tf",
        "nbshield",
        "thkcas",
        "tinstf",
        "dr_tf_wp",
        "wwp1",
        "wwp2",
        "casthi",
        # Possibly might not exist
        "tfthko",
        "beamwd",
        "rtanbeam",
        "beamwd",
    )
    (
        rmajor,
        rminor,
        rdewex,
        ddwex,
        n_tf,
        nbshield,
        thkcas,
        tinstf,
        dr_tf_wp,
        wwp1,
        wwp2,
        casthi,
        tfthko,
        beamwd,
        rtanbeam,
        beamwd,
    ) = attrgetter(attrs)(plot_data)

    # Check for Copper magnets
    i_tf_sup = (
        int(mfile_data.data["i_tf_sup"].get_scan(scan))
        if "i_tf_sup" in mfile_data.data
        else 1
    )

    i_tf_turns_integer = (
        int(mfile_data.data["i_tf_turns_integer"].get_scan(scan))
        if "i_tf_turns_integer" in mfile_data.data
        else 0
    )

    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_title("Toroidal cross-section")

    arc(axis, rmajor, style="dashed")

    data = (
        "ohcth",
        "precomp",
        "tfcth",
        "thshield_ib",
        "d_vv_in",
        "shldith",
        "blnkith",
        "fwith",
        "fwoth",
        "blnkoth",
        "shldoth",
        "d_vv_out",
        "thshield_ob",
    )
    # Colour in the main components
    for key in data:
        arc_fill(
            axis,
            *cumulative_radial_build(
                key, mfile_data, plot_data, scan, with_previous=True, verbose=False
            ),
            color=colour_dict[key],
        )

    arc_fill(axis, rmajor - rminor, rmajor + rminor, color=plasma)
    arc_fill(axis, rdewex, rdewex + ddwex, color=cryostat)

    # Segment the TF coil inboard
    # Calculate centrelines
    n = int(n_tf / 4) + 1
    spacing = 2 * np.pi / n_tf
    i = np.arange(0, n)

    ang = i * spacing
    angl = ang - spacing / 2
    angu = ang + spacing / 2
    r1 = cumulative_radial_build("gapoh", mfile_data, plot_data, scan, verbose=False)
    r2 = cumulative_radial_build("tfcth", mfile_data, plot_data, scan, verbose=False)
    r4, r3 = cumulative_radial_build(
        "tfthko", mfile_data, plot_data, scan, with_previous=True, verbose=False
    )

    for ang in (angl, angu):
        axis.plot(
            (r1 * np.cos(ang), r2 * np.cos(ang)),
            (r1 * np.sin(ang), r2 * np.sin(ang)),
            color="black",
        )

    # Annotate plot.
    axis.text(
        rmajor * np.cos(0.3),
        rmajor * np.sin(0.3),
        "plasma",
        fontsize=(12),
        ha="center",
        va="center",
    )
    axis.text(
        (rdewex + ddwex) / 1.41,
        (rdewex + ddwex) / 1.41,
        "cryostat",
        fontsize=(10),
        ha="left",
        va="bottom",
    )

    # Coil width
    w = r2 * np.tan(spacing / 2)
    for item in i:
        for ww, col in ((w + nbshield, nbshield), (w, "cyan")):
            # Neutral beam shielding and overlay TF coil segments
            TF_outboard(axis, item, n_tf=n_tf, r3=r3, r4=r4, w=ww, facecolor=col)

    # Winding pack : inboard (superconductor only)
    if i_tf_sup == 1:
        # Inboard
        x1 = r1 + thkcas + tinstf
        if i_tf_turns_integer == 1:
            axis.add_patch(
                patches.Rectangle([x1, 0], dr_tf_wp, wwp1 / 2, lw=0, facecolor=winding)
            )
        else:
            for i, wp in ((x1, wwp2 / 2), (x1 + dr_tf_wp / 2, wwp1 / 2)):
                axis.add_patch(
                    patches.Rectangle([i, 0], dr_tf_wp / 2, wp, lw=0, facecolor=winding)
                )

        # Outboard
        x1 = r3 + casthi + tinstf
        if i_tf_turns_integer == 1:
            axis.add_patch(
                patches.Rectangle([x1, 0], dr_tf_wp, wwp1 / 2, lw=0, facecolor=winding)
            )
        else:
            for i, wp in ((x1, wwp1 / 2), (x1 + dr_tf_wp / 2, wwp2 / 2)):
                axis.add_patch(
                    patches.Rectangle([i, 0], dr_tf_wp / 2, wp, lw=0, facecolor=winding)
                )

    iefrf = mfile_data.data["iefrf"].get_scan(scan)
    if (iefrf == 5) or (iefrf == 8):
        # Neutral beam geometry
        c = beamwd + 2 * nbshield
        e = np.sqrt(w**2 + (r3 + tfthko) ** 2)
        # Coordinates of the inner and outer edges of the beam at its tangency point
        rinner = rtanbeam - beamwd
        router = rtanbeam + beamwd
        beta = np.arccos(rinner / e)
        xouter = router * np.cos(beta)
        youter = router * np.sin(beta)
        # Corner of TF coils
        xcorner = r4
        ycorner = w + nbshield
        inner = ([rinner * np.cos(beta), xcorner], [rinner * np.sin(beta), ycorner])
        outer = (
            [xouter, xcorner + c * np.cos(beta) - nbshield * np.cos(beta)],
            [youter, ycorner + c * np.sin(beta) - nbshield * np.sin(beta)],
        )
        for x, y in (inner, outer):
            axis.plot(x, y, linestyle="dotted", color="black")

    # Ranges
    if demo_ranges:
        # DEMO : Fixed ranges for comparison
        axis.set_ylim([0, 20])
        axis.set_xlim([0, 20])
    else:
        # Adapatative ranges
        axis.set_ylim([0.0, axis.get_ylim()[1]])
        axis.set_xlim([0.0, axis.get_xlim()[1]])


def TF_outboard(axis, item, n_tf, r3, r4, w, facecolor):
    # with spacing
    ang = item * 2 * np.pi / n_tf
    dx = w * np.sin(ang)
    dy = w * np.cos(ang)
    verts = [
        (r * np.cos(ang) + x_s * dx, r * np.sin(ang) + y_s * dy)
        for x_s, y_s, r in zip((1, 1, -1 - 1), (-1, -1, 1, 1), (r3, r4, r4, r3))
    ]
    axis.add_patch(
        patches.PathPatch(Path(verts, closed=True), facecolor=facecolor, lw=0)
    )


def arc(axis, r, theta1=0, theta2=RTANGLE, style="solid"):
    """Plots an arc.

    Arguments

    axis: plot object
    r: radius
    theta1: starting polar angle
    theta2: finishing polar angle

    """
    ang = np.linspace(theta1, theta2)
    axis.plot(r * np.cos(ang), r * np.sin(ang), linestyle=style, color="black", lw=0.2)


def arc_fill(axis, r1, r2, color="pink"):
    """Fills the space between two quarter circles.

    Arguments

    axis: plot object
    r1, r2 radii to be filled

    """
    ang1 = np.linspace(0, RTANGLE, endpoint=True)
    ang2 = np.linspace(RTANGLE, 0, endpoint=True)
    verts = (
        list(zip(r1 * np.cos(ang1), r1 * np.sin(ang1)))
        + list(zip(r2 * np.cos(ang2), r2 * np.sin(ang2)))
        + [(r2, 0)]
    )
    axis.add_patch(patches.PathPatch(Path(verts, closed=True), facecolor=color, lw=0))


def ellips_fill(
    axis, a1=0, a2=0, b1=0, b2=0, x0=0, y0=0, ang1=0, ang2=RTANGLE, color="pink"
):
    """Fills the space between two concentric ellipse sectors.

    Arguments

    axis: plot object
    a1, a2, b1, b2 horizontal and vertical radii to be filled
    x0, y0 coordinates of centre of the ellipses
    ang1, ang2 are the polar angles of the start and end

    """
    angs = np.linspace(ang1, ang2, endpoint=True)
    r1 = ((np.cos(angs) / a1) ** 2 + (np.sin(angs) / b1) ** 2) ** (-0.5)
    xs1 = r1 * np.cos(angs) + x0
    ys1 = r1 * np.sin(angs) + y0
    angs = np.linspace(ang2, ang1, endpoint=True)
    r2 = ((np.cos(angs) / a2) ** 2 + (np.sin(angs) / b2) ** 2) ** (-0.5)
    xs2 = r2 * np.cos(angs) + x0
    ys2 = r2 * np.sin(angs) + y0
    verts = list(zip(xs1, ys1))
    verts.extend(list(zip(xs2, ys2)))
    endpoint = verts[-1:]
    verts.extend(endpoint)
    path = Path(verts, closed=True)
    patch = patches.PathPatch(path, facecolor=color, lw=0)
    axis.add_patch(patch)


def plot_nprofile(prof, demo_ranges, plot_data):
    """Function to plot density profile
    Arguments:
      prof --> axis object to add plot to
    """
    alphan = plot_data.alphan
    ne0 = plot_data.ne0
    neped = plot_data.neped
    nesep = plot_data.nesep
    rhopedn = plot_data.rhopedn
    ipedestal = plot_data.ipedestal

    prof.set_xlabel("r/a")
    prof.set_ylabel(r"$n_{e}\cdot 10^{19}$ $[\mathrm{m}^{-3}]$")
    prof.set_title("Density profile")

    if ipedestal == 1:
        rhocore1 = np.linspace(0, 0.95 * rhopedn)
        rhocore2 = np.linspace(0.95 * rhopedn, rhopedn)
        rhocore = np.append(rhocore1, rhocore2)
        ncore = neped + (ne0 - neped) * (1 - rhocore**2 / rhopedn**2) ** alphan

        rhosep = np.linspace(rhopedn, 1)
        nsep = nesep + (neped - nesep) * (1 - rhosep) / (1 - min(0.9999, rhopedn))

        rho = np.append(rhocore, rhosep)
        ne = np.append(ncore, nsep)
    else:
        rho1 = np.linspace(0, 0.95)
        rho2 = np.linspace(0.95, 1)
        rho = np.append(rho1, rho2)
        ne = ne0 * (1 - rho**2) ** alphan
    ne = ne / 1e19
    prof.plot(rho, ne)

    # Ranges
    prof.set_xlim([0, 1])
    if demo_ranges:
        # DEMO : Fixed ranges for comparison
        prof.set_ylim([0, 20])
    else:
        # Adaptive ranges
        prof.set_ylim([0, prof.get_ylim()[1]])


def plot_tprofile(prof, demo_ranges, plot_data):
    """Function to plot temperature profile
    Arguments:
      prof --> axis object to add plot to
    """
    alphat = plot_data.alphat
    rhopedt = plot_data.rhopedt
    tbeta = plot_data.tbeta
    te0 = plot_data.te0
    teped = plot_data.teped
    tesep = plot_data.tesep
    ipedestal = plot_data.ipedestal

    prof.set_xlabel("r/a")
    prof.set_ylabel("$T_{e}$ [keV]")
    prof.set_title("Temperature profile")

    if ipedestal == 1:
        rhocore1 = np.linspace(0, 0.9 * rhopedt)
        rhocore2 = np.linspace(0.9 * rhopedt, rhopedt)
        rhocore = np.append(rhocore1, rhocore2)
        tcore = teped + (te0 - teped) * (1 - (rhocore / rhopedt) ** tbeta) ** alphat

        rhosep = np.linspace(rhopedt, 1)
        tsep = tesep + (teped - tesep) * (1 - rhosep) / (1 - min(0.9999, rhopedt))

        rho = np.append(rhocore, rhosep)
        te = np.append(tcore, tsep)
    else:
        rho1 = np.linspace(0, 0.95)
        rho2 = np.linspace(0.95, 1)
        rho = np.append(rho1, rho2)
        te = te0 * (1 - rho**2) ** alphat
    prof.plot(rho, te)

    # Ranges
    # ---
    prof.set_xlim([0, 1])
    # DEMO : Fixed ranges for comparison
    if demo_ranges:
        prof.set_ylim([0, 50])

    # Adapatative ranges
    else:
        prof.set_ylim([0, prof.get_ylim()[1]])
    # ---


def plot_qprofile(prof, demo_ranges, plot_data):
    """Function to plot q profile, formula taken from Nevins bootstrap model.

    Arguments:
      prof --> axis object to add plot to
    """
    q0 = plot_data.q0
    q95 = plot_data.q95

    prof.set_xlabel("r/a")
    prof.set_ylabel("q(r)")
    prof.set_title("q profile")

    rho = np.linspace(0, 1)
    q_r_nevin = q0 + (q95 - q0) * (rho + rho * rho + rho**3) / (3.0)
    q_r_sauter = q0 + (q95 - q0) * (rho * rho)

    prof.plot(rho, q_r_nevin, label="Nevins")
    prof.plot(rho, q_r_sauter, label="Sauter")
    prof.legend()

    # Ranges
    # ---
    prof.set_xlim([0, 1])
    # DEMO : Fixed ranges for comparison
    if demo_ranges:
        prof.set_ylim([0, 10])

    # Adapatative ranges
    else:
        prof.set_ylim([0, prof.get_ylim()[1]])
    # ---


def read_imprad_data(skiprows, data_path):
    """Function to read all data needed for creation of radiation profile

    Arguments:
        skiprows --> number of rows to skip when reading impurity data files
        data_path --> path to impurity data
    """
    label = [
        "H_",
        "He",
        "Be",
        "C_",
        "N_",
        "O_",
        "Ne",
        "Si",
        "Ar",
        "Fe",
        "Ni",
        "Kr",
        "Xe",
        "W_",
    ]
    lzdata = [0.0 for x in range(len(label))]
    # DATAFILENAME = p DATAPATH +

    for i in range(len(label)):
        file_iden = data_path + label[i].ljust(3, "_")

        Te = None
        lz = None
        zav = None

        for header in read_impurity_file(file_iden + "lz_tau.dat"):
            if "Te[eV]" in header.content:
                Te = np.asarray(header.data, dtype=float)

            if "infinite confinement" in header.content:
                lz = np.asarray(header.data, dtype=float)
        for header in read_impurity_file(file_iden + "z_tau.dat"):
            if "infinite confinement" in header.content:
                zav = np.asarray(header.data, dtype=float)

        lzdata[i] = np.column_stack((Te, lz, zav))

    # then switch string to floats
    impdata = np.array(lzdata, dtype=float)
    return impdata


def synchrotron_rad(plot_data):
    """Function for Synchrotron radiation power calculation from Albajar, Nuclear Fusion 41 (2001) 665
      Fidone, Giruzzi, Granata, Nuclear Fusion 41 (2001) 1755

    Arguments:
    """
    # tbet is betaT in Albajar, not to be confused with plasma beta
    vol = plot_data.vol
    rmajor = plot_data.rmajor
    rminor = plot_data.rminor
    alphan = plot_data.alphan
    alphat = plot_data.alphat
    ne0 = plot_data.ne0
    te0 = plot_data.te0
    ssync = plot_data.ssync
    bt = plot_data.bt

    tbet = 2.0
    # rpow is the(1-Rsyn) power dependence based on plasma shape
    # (see Fidone)
    rpow = 0.62
    kap = vol / (2.0 * 3.1415**2 * rmajor * rminor**2)

    # No account is taken of pedestal profiles here, other than use of
    # the correct ne0 and te0...
    de2o = 1.0e-20 * ne0
    pao = 6.04e3 * (rminor * de2o) / bt
    gfun = 0.93 * (1.0 + 0.85 * np.exp(-0.82 * rmajor / rminor))
    kfun = (alphan + 3.87e0 * alphat + 1.46) ** (-0.79)
    kfun = kfun * (1.98 + alphat) ** 1.36 * tbet**2.14
    kfun = kfun * (tbet**1.53 + 1.87 * alphat - 0.16) ** (-1.33)
    dum = 1.0 + 0.12 * (te0 / (pao**0.41)) * (1.0 - ssync) ** 0.41
    # Very high T modification, from Fidone
    dum = dum ** (-1.51)

    psync = 3.84e-8 * (1.0e0 - ssync) ** rpow * rmajor * rminor**1.38
    psync = psync * kap**0.79 * bt**2.62 * de2o**0.38
    psync = psync * te0 * (16.0 + te0) ** 2.61 * dum * gfun * kfun

    # psyncpv should be per unit volume
    # Albajar gives it as total
    psyncpv = psync / vol
    print("psyncpv = ", psyncpv * vol)  # matches the out.dat file

    return psyncpv


def plot_radprofile(prof, mfile_data, plot_data, scan, impp, demo_ranges) -> float:
    """Function to plot radiation profile, formula taken from ???.

    Arguments:
      prof --> axis object to add plot to
      mfile_data --> MFILE.DAT object
      scan --> scan number to use
      impp --> impurity path
    """
    alphan = plot_data.alphan
    alphat = plot_data.alphat
    ne0 = plot_data.ne0
    neped = plot_data.neped
    nesep = plot_data.nesep
    rhopedn = plot_data.rhopedn
    rhopedt = plot_data.rhopedt
    tbeta = plot_data.tbeta
    te0 = plot_data.te0
    teped = plot_data.teped
    tesep = plot_data.tesep
    ipedestal = plot_data.ipedestal

    prof.set_xlabel("r/a")
    prof.set_ylabel(r"$P_{\mathrm{rad}}$ $[\mathrm{MW.m}^{-3}]$")
    prof.set_title("Radiation profile")

    # read in the impurity data
    imp_data = read_imprad_data(2, impp)

    # find impurity densities
    imp_frac = np.array(
        [
            mfile_data.data["fimp(01)"].get_scan(scan),
            mfile_data.data["fimp(02)"].get_scan(scan),
            mfile_data.data["fimp(03)"].get_scan(scan),
            mfile_data.data["fimp(04)"].get_scan(scan),
            mfile_data.data["fimp(05)"].get_scan(scan),
            mfile_data.data["fimp(06)"].get_scan(scan),
            mfile_data.data["fimp(07)"].get_scan(scan),
            mfile_data.data["fimp(08)"].get_scan(scan),
            mfile_data.data["fimp(09)"].get_scan(scan),
            mfile_data.data["fimp(10)"].get_scan(scan),
            mfile_data.data["fimp(11)"].get_scan(scan),
            mfile_data.data["fimp(12)"].get_scan(scan),
            mfile_data.data["fimp(13)"].get_scan(scan),
            mfile_data.data["fimp(14)"].get_scan(scan),
        ]
    )

    if ipedestal == 0:
        # Intialise the radius
        rho = np.linspace(0, 1.0)

        # The density profile
        ne = ne0 * (1 - rho**2) ** alphan

        # The temperature profile
        te = te0 * (1 - rho**2) ** alphat

    if ipedestal == 1:
        # Intialise the normalised radius
        rhoped = (rhopedn + rhopedt) / 2.0
        rhocore1 = np.linspace(0, 0.95 * rhoped)
        rhocore2 = np.linspace(0.95 * rhoped, rhoped)
        rhocore = np.append(rhocore1, rhocore2)
        rhosep = np.linspace(rhoped, 1)
        rho = np.append(rhocore, rhosep)

        # The density and temperature profile
        # done in such away as to allow for plotting pedestals
        # with different rhopedn and rhopedt
        ne = np.zeros(rho.shape[0])
        te = np.zeros(rho.shape[0])
        for q in range(rho.shape[0]):
            if rho[q] <= rhopedn:
                ne[q] = (
                    neped + (ne0 - neped) * (1 - rho[q] ** 2 / rhopedn**2) ** alphan
                )
            else:
                ne[q] = nesep + (neped - nesep) * (1 - rho[q]) / (
                    1 - min(0.9999, rhopedn)
                )

            if rho[q] <= rhopedt:
                te[q] = (
                    teped + (te0 - teped) * (1 - (rho[q] / rhopedt) ** tbeta) ** alphat
                )
            else:
                te[q] = tesep + (teped - tesep) * (1 - rho[q]) / (
                    1 - min(0.9999, rhopedt)
                )

        # ncore = neped + (ne0-neped) * (1-rhocore**2/rhopedn**2)**alphan
        # nsep = nesep + (neped-nesep) * (1-rhosep)/(1-min(0.9999, rhopedn))
        # ne = np.append(ncore, nsep)

        # The temperatue profile
        # tcore = teped + (te0-teped) * (1-(rhocore/rhopedt)**tbeta)**alphat
        # tsep = tesep + (teped-tesep)* (1-rhosep)/(1-min(0.9999,rhopedt))
        # te = np.append(tcore,tsep)

    # Intailise the radiation profile arrays
    pimpden = np.zeros([imp_data.shape[0], te.shape[0]])
    lz = np.zeros([imp_data.shape[0], te.shape[0]])
    prad = np.zeros(te.shape[0])

    # psyncpv = synchrotron_rad()

    # Intailise the impurity radiation profile
    for k in range(te.shape[0]):
        for i in range(imp_data.shape[0]):
            if te[k] <= imp_data[i][0][0]:
                lz[i][k] = imp_data[i][0][1]
            elif te[k] >= imp_data[i][imp_data.shape[1] - 1][0]:
                lz[i][k] = imp_data[i][imp_data.shape[1] - 1][1]
            else:
                for j in range(imp_data.shape[1] - 1):
                    # Linear interpolation in log-log space
                    if (te[k] > imp_data[i][j][0]) and (te[k] <= imp_data[i][j + 1][0]):
                        yi = np.log(imp_data[i][j][1])
                        xi = np.log(imp_data[i][j][0])
                        c = (np.log(imp_data[i][j + 1][1]) - yi) / (
                            np.log(imp_data[i][j + 1][0]) - xi
                        )
                        lz[i][k] = np.exp(yi + c * (np.log(te[k]) - xi))
                        # Zav[i][k] = imp_data[i][j][2]
            # The impurity radiation
            pimpden[i][k] = imp_frac[i] * ne[k] * ne[k] * lz[i][k]
            # The Bremsstrahlung
            # pbremden[i][k] = imp_frac[i] * ne[k] * ne[k] * Zav[i][k] * Zav[i][k] * 5.355e-37 * np.sqrt(te[k])

        for l in range(imp_data.shape[0]):  # noqa: E741
            prad[k] = prad[k] + pimpden[l][k] * 2.0e-6
            # pbrem[k] = pbrem[k] + pbremden[l][k] * 2.0e-6

    # benchmark prad again outfile so mod prad
    # pbremint = (rho[1:] * pbrem[1:]) @ drho
    # pradint = prad[1:] @ drho * 2.0e-5
    # pbremint = pbrem[1:] @ drho * 2.0e-5

    # print('prad = ',prad)
    # print('pbrem = ',pbrem)
    # print(1.0e32*lz[12])
    # print('pradpv = ',pradint)
    # print('pbrempv = ',pbremint)
    # print('pbremmw = ',pbremint*vol)
    # print('pradmw = ', pradint*vol, 'MW') # pimp = pline + pbrem

    prof.plot(rho, prad, label="Total")
    prof.plot(rho, pimpden[0] * 2.0e-6, label="H")
    prof.plot(rho, pimpden[1] * 2.0e-6, label="He")
    labels = ["Be", "C", "N", "O", "Ne", "Si", "Ar", "Fe", "Ni", "Kr", "Xe", "W"]
    for i_f, label, p_den in zip(imp_frac[2:14], labels, pimpden[2:14]):
        if i_f > 1.0e-30:
            prof.plot(rho, p_den * 2.0e-6, label=label)
    prof.legend()

    # Ranges
    prof.set_xlim([0, 1])
    if demo_ranges:
        # DEMO : Fixed ranges for comparison
        prof.set_ylim([0, 0.5])
    else:
        # Adapatative ranges
        prof.set_ylim([0, prof.get_ylim()[1]])


def plot_vacuum_vessel(axis, mfile_data, plot_data, scan):
    """Function to plot vacuum vessel

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE data object
        scan --> scan number to use
    """
    cumulative_upper = plot_data.cumulative_upper
    cumulative_lower = plot_data.cumulative_lower
    upper = plot_data.upper
    lower = plot_data.lower

    i_single_null = mfile_data.data["i_single_null"].get_scan(scan)
    triang = mfile_data.data["triang95"].get_scan(scan)
    temp_array_1 = ()
    temp_array_2 = ()

    # Outer side (furthest from plasma)
    radx = (
        cumulative_radial_build("d_vv_out", mfile_data, plot_data, scan)
        + cumulative_radial_build("gapds", mfile_data, plot_data, scan)
    ) / 2.0
    rminx = (
        cumulative_radial_build("d_vv_out", mfile_data, plot_data, scan)
        - cumulative_radial_build("gapds", mfile_data, plot_data, scan)
    ) / 2.0

    kapx = cumulative_upper["d_vv_top"] / rminx

    if i_single_null == 1:
        (rs, zs) = plotdh(axis, radx, rminx, triang, kapx)
        temp_array_1 = (*temp_array_1, rs, zs)

    kapx = cumulative_lower["d_vv_bot"] / rminx
    (rs, zs) = plotdh(axis, radx, rminx, triang, kapx)
    temp_array_2 = (*temp_array_2, rs, zs)

    # Inner side (nearest to the plasma)
    radx = (
        cumulative_radial_build("shldoth", mfile_data, plot_data, scan)
        + cumulative_radial_build("d_vv_in", mfile_data, plot_data, scan)
    ) / 2.0
    rminx = (
        cumulative_radial_build("shldoth", mfile_data, plot_data, scan)
        - cumulative_radial_build("d_vv_in", mfile_data, plot_data, scan)
    ) / 2.0

    if i_single_null == 1:
        kapx = (cumulative_upper["d_vv_top"] - upper["d_vv_top"]) / rminx
        (rs, zs) = plotdh(axis, radx, rminx, triang, kapx)
        temp_array_1 = (*temp_array_1, rs, zs)

    kapx = (cumulative_lower["d_vv_bot"] + lower["d_vv_bot"]) / rminx
    (rs, zs) = plotdh(axis, radx, rminx, triang, kapx)
    temp_array_2 = (*temp_array_2, rs, zs)

    # Single null: Draw top half from output
    # Double null: Reflect bottom half to top
    if i_single_null == 1:
        rs = np.concatenate([temp_array_1[0], temp_array_1[2][::-1]])
        zs = np.concatenate([temp_array_1[1], temp_array_1[3][::-1]])
        axis.fill(rs, zs, color=vessel)

    rs = np.concatenate([temp_array_2[0], temp_array_2[2][::-1]])
    zs = np.concatenate([temp_array_2[1], temp_array_2[3][::-1]])
    axis.fill(rs, zs, color=vessel)
    # For double null, reflect shape of lower half to top instead
    if i_single_null == 0:
        axis.fill(rs, -zs, color=vessel)


def plot_shield(axis, mfile_data, plot_data, scan):
    """Function to plot shield

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE data object
        scan --> scan number to use
    """
    cumulative_upper = plot_data.cumulative_upper
    cumulative_lower = plot_data.cumulative_lower
    i_single_null = mfile_data.data["i_single_null"].get_scan(scan)
    triang = mfile_data.data["triang95"].get_scan(scan)

    # Side furthest from plasma
    radx = (
        cumulative_radial_build("shldoth", mfile_data, plot_data, scan)
        + cumulative_radial_build("d_vv_in", mfile_data, plot_data, scan)
    ) / 2.0
    rminx = (
        cumulative_radial_build("shldoth", mfile_data, plot_data, scan)
        - cumulative_radial_build("d_vv_in", mfile_data, plot_data, scan)
    ) / 2.0

    if i_single_null == 1:
        kapx = cumulative_upper["shldtth"] / rminx
        temp_array_1 = plotdh(axis, radx, rminx, triang, kapx)

    kapx = cumulative_lower["shldlth"] / rminx
    temp_array_2 = plotdh(axis, radx, rminx, triang, kapx)

    # Side nearest to the plasma
    radx = (
        cumulative_radial_build("vvblgapo", mfile_data, plot_data, scan)
        + cumulative_radial_build("shldith", mfile_data, plot_data, scan)
    ) / 2.0
    rminx = (
        cumulative_radial_build("vvblgapo", mfile_data, plot_data, scan)
        - cumulative_radial_build("shldith", mfile_data, plot_data, scan)
    ) / 2.0

    # Single null: Draw top half from output
    # Double null: Reflect bottom half to top
    if i_single_null == 1:
        kapx = (cumulative_upper["vvblgap"]) / rminx
        temp_array_1 = (*temp_array_1, *plotdh(axis, radx, rminx, triang, kapx))
        rs = np.concatenate([temp_array_1[0], temp_array_1[2][::-1]])
        zs = np.concatenate([temp_array_1[1], temp_array_1[3][::-1]])
        axis.fill(rs, zs, color=shield)

    kapx = (cumulative_lower["divfix"]) / rminx
    temp_array_2 = (*temp_array_2, *plotdh(axis, radx, rminx, triang, kapx))
    rs = np.concatenate([temp_array_2[0], temp_array_2[2][::-1]])
    zs = np.concatenate([temp_array_2[1], temp_array_2[3][::-1]])
    axis.fill(rs, zs, color=shield)
    if i_single_null == 0:
        axis.fill(rs, -zs, color=shield)


def plot_blanket(axis, mfile_data, plot_data, scan):
    """Function to plot blanket

    Arguments:
      axis --> axis object to plot to
      mfile_data --> MFILE.DAT object
      scan --> scan number to use

    """
    cumulative_upper = plot_data.cumulative_upper
    cumulative_lower = plot_data.cumulative_lower
    triang = plot_data.triang
    blnkoth = plot_data.blnkoth
    blnkith = plot_data.blnkith

    # Single null: Draw top half from output
    # Double null: Reflect bottom half to top
    i_single_null = plot_data.i_single_null
    if i_single_null == 1:
        # Upper blanket: outer surface
        radx = (
            cumulative_radial_build("blnkoth", mfile_data, plot_data, scan)
            + cumulative_radial_build("vvblgapi", mfile_data, plot_data, scan)
        ) / 2.0
        rminx = (
            cumulative_radial_build("blnkoth", mfile_data, plot_data, scan)
            - cumulative_radial_build("vvblgapi", mfile_data, plot_data, scan)
        ) / 2.0

        kapx = cumulative_upper["blnktth"] / rminx
        point_array = plotdh(axis, radx, rminx, triang, kapx)

        # Upper blanket: inner surface
        radx = (
            cumulative_radial_build("fwoth", mfile_data, plot_data, scan)
            + cumulative_radial_build("blnkith", mfile_data, plot_data, scan)
        ) / 2.0
        rminx = (
            cumulative_radial_build("fwoth", mfile_data, plot_data, scan)
            - cumulative_radial_build("blnkith", mfile_data, plot_data, scan)
        ) / 2.0

        kapx = cumulative_upper["fwtth"] / rminx
        point_array = (*point_array, *plotdh(axis, radx, rminx, triang, kapx))

        # Plot upper blanket
        rs = np.concatenate([point_array[0], point_array[2][::-1]])
        zs = np.concatenate([point_array[1], point_array[3][::-1]])
        axis.fill(rs, zs, color=blanket)

    # Lower blanket
    blnktth = mfile_data.data["blnktth"].get_scan(scan)
    c_shldith = cumulative_radial_build("shldith", mfile_data, plot_data, scan)
    c_blnkoth = cumulative_radial_build("blnkoth", mfile_data, plot_data, scan)
    divgap = cumulative_lower["divfix"]
    plotdhgap(
        axis, c_shldith, c_blnkoth, blnkith, blnkoth, divgap, -blnktth, triang, blanket
    )
    if i_single_null == 0:
        plotdhgap(
            axis,
            c_shldith,
            c_blnkoth,
            blnkith,
            blnkoth,
            -divgap,
            -blnktth,
            triang,
            blanket,
        )


def plot_firstwall(axis, mfile_data, plot_data, scan):
    """Function to plot first wall

    Arguments:
      axis --> axis object to plot to
      mfile_data --> MFILE.DAT object
      scan --> scan number to use

    """
    cumulative_upper = plot_data.cumulative_upper
    cumulative_lower = plot_data.cumulative_lower
    triang = plot_data.triang
    fwoth = plot_data.fwoth
    fwith = plot_data.fwith
    blnktth = plot_data.blnktth
    tfwvt = plot_data.fwtth
    i_single_null = plot_data.i_single_null

    # Single null: Draw top half from output
    # Double null: Reflect bottom half to top
    if i_single_null == 1:
        # Upper first wall: outer surface
        radx = (
            cumulative_radial_build("fwoth", mfile_data, plot_data, scan)
            + cumulative_radial_build("blnkith", mfile_data, plot_data, scan)
        ) / 2.0
        rminx = (
            cumulative_radial_build("fwoth", mfile_data, plot_data, scan)
            - cumulative_radial_build("blnkith", mfile_data, plot_data, scan)
        ) / 2.0

        kapx = cumulative_upper["fwtth"] / rminx
        point_array = plotdh(axis, radx, rminx, triang, kapx)

        # Upper first wall: inner surface
        radx = (
            cumulative_radial_build("scraplo", mfile_data, plot_data, scan)
            + cumulative_radial_build("fwith", mfile_data, plot_data, scan)
        ) / 2.0
        rminx = (
            cumulative_radial_build("scraplo", mfile_data, plot_data, scan)
            - cumulative_radial_build("fwith", mfile_data, plot_data, scan)
        ) / 2.0

        point_array = (*point_array, *plotdh(axis, radx, rminx, triang, kapx))

        # Plot upper first wall
        rs = np.concatenate([point_array[0], point_array[2][::-1]])
        zs = np.concatenate([point_array[1], point_array[3][::-1]])
        axis.fill(rs, zs, color=firstwall)

    # Lower first wall
    c_blnkith = cumulative_radial_build("blnkith", mfile_data, plot_data, scan)
    c_fwoth = cumulative_radial_build("fwoth", mfile_data, plot_data, scan)
    divgap = cumulative_lower["divfix"]

    plotdhgap(
        axis,
        c_blnkith,
        c_fwoth,
        fwith,
        fwoth,
        divgap + blnktth,
        -tfwvt,
        triang,
        firstwall,
    )

    if i_single_null == 0:
        plotdhgap(
            axis,
            c_blnkith,
            c_fwoth,
            fwith,
            fwoth,
            -(divgap + blnktth),
            -tfwvt,
            triang,
            firstwall,
        )


def angle_check(angle1, angle2):
    """Function to perform TF coil angle check"""
    if angle1 > 1:
        angle1 = 1
    elif angle1 < -1:
        angle1 = -1
    if angle2 > 1:
        angle2 = 1
    elif angle2 < -1:
        angle2 = -1
    return angle1, angle2


def plot_tf_coils(axis, mfile_data, plot_data, scan):
    """Function to plot TF coils

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """

    tfcth = plot_data.tfcth
    rt = RTANGLE
    rt2 = 2 * rt

    # Arc points
    # MDK Only 4 points now required for elliptical arcs
    x1, x2, x3, x4, x5 = (
        mfile_data.data[f"xarc({i})"].get_scan(scan) for i in range(1, 6)
    )
    y1, y2, y3, y4, y5 = (
        mfile_data.data[f"xarc({i})"].get_scan(scan) for i in range(1, 6)
    )
    if y3 != 0:
        print("TF coil geometry: The value of yarc(3) is not zero, but should be.")

    # Check for TF coil shape
    i_tf_shape = (
        int(mfile_data.data["i_tf_shape"].get_scan(scan))
        if "i_tf_shape" in mfile_data.data
        else 1
    )

    #  D-shaped (i_tf_shape=1), Picture frame (i_tf_shape=2)
    if i_tf_shape == 2:
        # Inboard leg
        axis.add_patch(
            patches.Rectangle(
                [x5 - tfcth, y5 - tfcth],
                tfcth,
                (y1 - y5 + 2.0 * tfcth),
                lw=0,
                facecolor="cyan",
            )
        )
        # Outboard leg vertical
        axis.add_patch(
            patches.Rectangle(
                [x4, y4 - tfcth], tfcth, (y2 - y4 + 2.0 * tfcth), lw=0, facecolor="cyan"
            )
        )
        # Outboard leg horizontal bottom
        axis.add_patch(
            patches.Rectangle([x5, y5 - tfcth], x4 - x5, tfcth, lw=0, facecolor="cyan")
        )
        # Outboard leg horizontal top
        axis.add_patch(
            patches.Rectangle([x1, y1], x2 - x1, tfcth, lw=0, facecolor="cyan")
        )

    else:
        # Inboard upper arc
        c = "cyan"
        a1 = x2 - x1
        b1 = y2 - y1
        ellips_fill(
            axis,
            a1=a1,
            a2=a1 + tfcth,
            b1=b1,
            b2=b1 + tfcth,
            x0=x2,
            y0=y1,
            ang1=rt,
            ang2=rt2,
            color=c,
        )
        # Outboard upper arc
        a1 = x3 - x2
        ellips_fill(
            axis,
            a1=a1,
            a2=a1 + tfcth,
            b1=y2,
            b2=y2 + tfcth,
            x0=x2,
            y0=0,
            ang1=0,
            ang2=rt,
            color=c,
        )
        # Inboard lower arc
        a1 = x4 - x5
        b1 = y5 - y4
        ellips_fill(
            axis,
            a1=a1,
            a2=a1 + tfcth,
            b1=b1,
            b2=b1 + tfcth,
            x0=x4,
            y0=y5,
            ang1=-rt,
            ang2=-rt2,
            color=c,
        )
        # Outboard lower arc
        a1 = x3 - x2
        ellips_fill(
            axis,
            a1=a1,
            a2=a1 + tfcth,
            b1=-y4,
            b2=tfcth - y4,
            x0=x4,
            y0=0,
            ang1=0,
            ang2=-rt,
            color=c,
        )
        # Vertical leg
        # Bottom left corner
        axis.add_patch(
            patches.Rectangle([x5 - tfcth, y5], tfcth, (y1 - y5), lw=0, facecolor=c)
        )


def plot_pf_coils(axis, mfile_data, scan):
    """Function to plot PF coils

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use
    """
    # Number of coils (1 is OH coil)
    number_of_coils = 0
    for item in mfile_data.data:
        if "rpf[" in item:
            number_of_coils += 1

    bore = mfile_data.data["bore"].get_scan(scan)
    ohcth = mfile_data.data["ohcth"].get_scan(scan)
    ohdz = mfile_data.data["ohdz"].get_scan(scan)

    # Check for Central Solenoid
    iohcl = mfile_data.data["iohcl"].get_scan(scan) if "iohcl" in mfile_data.data else 1

    # If Central Solenoid present, ignore last entry in for loop
    # The last entry will be the OH coil in this case
    for coil in range(number_of_coils + 1 if iohcl == 0 else number_of_coils):
        r = mfile_data.data[f"rpf[{coil:01}]"].get_scan(scan)
        z = mfile_data.data[f"zpf[{coil:01}]"].get_scan(scan)
        dr = mfile_data.data[f"pfdr[{coil:01}]"].get_scan(scan)
        dz = mfile_data.data[f"pfdz[{coil:01}]"].get_scan(scan)
        r_points = [r + val * 0.5 * dr for val in (-1, -1, +1, +1, -1)]
        z_points = [r + val * 0.5 * dr for val in (-1, +1, +1, -1, -1)]
        axis.plot(r_points, z_points, color="black")
        axis.add_patch(
            patches.Rectangle([bore, -ohdz / 2], ohcth, ohdz, lw=0, facecolor="pink")
        )
        axis.text(
            r, z, str(coil + 1), ha="center", va="center", fontsize=5 * abs(dr * dz)
        )


def plot_info(axis, data, mfile_data, scan):
    """Function to plot data in written form on a matplotlib plot.

    Arguments:
        axis --> axis object to plot to
        data --> plot information
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    eqpos = 0.7
    for i, d in enumerate(data):
        col = "black"
        if mfile_data.data[d[0]].exists:
            if mfile_data.data[d[0]].var_flag == "ITV":
                col = "red"
            elif mfile_data.data[d[0]].var_flag == "OP":
                col = "blue"
        axis.text(0, -i, d[1], color=col, ha="left", va="center")
        if isinstance(d[0], str):
            if d[0] == "":
                axis.text(eqpos, -i, "\n", ha="left", va="center")
            elif d[0][0] == "#":
                axis.text(-0.05, -i, f"{d[0][1:]}\n", ha="left", va="center")
            elif d[0][0] == "!":
                value = str(d[0][1:]).replace('"', "")
                axis.text(0.4, -i, f"-->  {value} {d[2]}", ha="left", va="center")
            elif mfile_data.data[d[0]].exists:
                dat = mfile_data.data[d[0]].get_scan(scan)
                value = (
                    dat
                    if isinstance(dat, str)
                    else f"{mfile_data.data[d[0]].get_scan(scan):.4g}"
                )
                if "alpha" in d[0]:
                    value = str(float(value) + 1.0)
                axis.text(
                    eqpos, -i, f"= {value} {d[2]}", color=col, ha="left", va="center"
                )
            else:
                mfile_data.data[d[0]].get_scan(-1)
                axis.text(
                    eqpos, -i, "=ERROR! Var missing", color=col, ha="left", va="center"
                )
        else:
            dat = d[0] if isinstance(d[0], str) else f"{d[0]:.4g}"
            axis.text(eqpos, -i, f"= {dat} {d[2]}", color=col, ha="left", va="center")


def plot_header(axis, mfile_data, scan):
    """Function to plot header info: date, rutitle etc

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    # Load dicts from dicts JSON file
    dicts = get_dicts()

    axis.set_ylim([-16, 1])
    axis.set_xlim([0, 1])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)
    minmax = str(abs(int(mfile_data.data["minmax"].get_scan(-1))))

    data2 = [
        (f"!{mfile_data.data['runtitle'].get_scan(-1)!s}", "Run title", ""),
        (f"!{mfile_data.data['procver'].get_scan(-1)!s}", "PROCESS Version", ""),
        (f"!{mfile_data.data['date'].get_scan(-1)}", "Date:", ""),
        (f"!{mfile_data.data['time'].get_scan(-1)}", "Time:", ""),
        (f"!{mfile_data.data['username'].get_scan(-1)}", "User:", ""),
        (f"!{dicts['DICT_OPTIMISATION_VARS'][minmax]}", "Optimising:", ""),
    ]

    fimp_data = {
        "D + T": mfile_data.data["fimp(01)"].get_scan(scan),
        "He": mfile_data.data["fimp(02)"].get_scan(scan),
        "Be": mfile_data.data["fimp(03)"].get_scan(scan),
        "C": mfile_data.data["fimp(04)"].get_scan(scan),
        "N": mfile_data.data["fimp(05)"].get_scan(scan),
        "O": mfile_data.data["fimp(06)"].get_scan(scan),
        "Ne": mfile_data.data["fimp(07)"].get_scan(scan),
        "Si": mfile_data.data["fimp(08)"].get_scan(scan),
        "Ar": mfile_data.data["fimp(09)"].get_scan(scan),
        "Fe": mfile_data.data["fimp(10)"].get_scan(scan),
        "Ni": mfile_data.data["fimp(11)"].get_scan(scan),
        "Kr": mfile_data.data["fimp(12)"].get_scan(scan),
        "Xe": mfile_data.data["fimp(13)"].get_scan(scan),
        "W": mfile_data.data["fimp(14)"].get_scan(scan),
    }

    data = [("", "", ""), ("", "", "")]

    for no, (name, imp) in enumerate(fimp_data.items()):
        if no < 2 or imp > 1e-10:
            data += [(imp, name, "")]

    if len(data) - 2 > 11:
        data = [("", "", ""), ("", "", ""), ("", "More than 11 impurities", "")]
    else:
        axis.text(-0.05, -6.4, "Plasma composition:", ha="left", va="center")
        axis.text(
            -0.05,
            -7.2,
            "Number densities relative to electron density:",
            ha="left",
            va="center",
        )
    data2 += data

    axis.text(-0.05, -12.6, "Colour Legend:", ha="left", va="center")
    axis.text(0.0, -13.4, "ITR", color="red", ha="left", va="center")
    axis.text(0.0, -14.2, "OP", color="blue", ha="left", va="center")

    plot_info(axis, data2, mfile_data, scan)


def plot_geometry_info(axis, mfile_data, plot_data, scan):
    """Function to plot geometry info

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    axis.text(-0.05, 1, "Geometry:", ha="left", va="center")
    axis.set_ylim([-16, 1])
    axis.set_xlim([0, 1])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)

    data = [
        ("rmajor", "$R_0$", "m"),
        ("rminor", "a", "m"),
        ("aspect", "A", ""),
        ("kappa95", r"$\kappa_{95}$", ""),
        ("triang95", r"$\delta_{95}$", ""),
        ("sarea", "Surface area", "m$^2$"),
        ("vol", "Plasma volume", "m$^3$"),
        ("n_tf", "No. of TF coils", ""),
        (plot_data.shldith + plot_data.blnkith, "inboard blanket+shield", "m"),
        (plot_data.shldoth + plot_data.blnkoth, "ouboard blanket+shield", "m"),
        ("powfmw", "Fusion power", "MW"),
        ("bigq", "$Q$", ""),
        ("", "", ""),
    ]

    plot_info(axis, data, mfile_data, scan)


def plot_physics_info(axis, mfile_data, scan):
    """Function to plot geometry info

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    axis.text(-0.05, 1, "Physics:", ha="left", va="center")
    axis.set_ylim([-16, 1])
    axis.set_xlim([0, 1])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)

    nong = mfile_data.data["dnla"].get_scan(scan) / mfile_data.data[
        "dlimit(7)"
    ].get_scan(scan)
    dene = mfile_data.data["dene"].get_scan(scan)
    dnz = mfile_data.data["dnz"].get_scan(scan) / dene

    tepeak = mfile_data.data["te0"].get_scan(scan) / mfile_data.data["te"].get_scan(scan)
    nepeak = mfile_data.data["ne0"].get_scan(scan) / dene

    # Assume Martin scaling if pthresh is not printed
    # Accounts for pthresh not being written prior to issue #679 and #680
    pthresh = mfile_data.data[
        "plhthresh" if "plhthresh" in mfile_data.data else "pthrmw(6)"
    ].get_scan(scan)

    data = [
        ("plascur/1d6", "$I_p$", "MA"),
        ("bt", "Vacuum $B_T$ at $R_0$", "T"),
        ("q95", r"$q_{\mathrm{95}}$", ""),
        ("normalised_thermal_beta", r"$\beta_N$, thermal", "% m T MA$^{-1}$"),
        ("normalised_toroidal_beta", r"$\beta_N$, toroidal", "% m T MA$^{-1}$"),
        ("thermal_poloidal_beta", r"$\beta_P$, thermal", ""),
        ("betap", r"$\beta_P$, total", ""),
        ("te", r"$< t_e >$", "keV"),
        ("dene", r"$< n_e >$", "m$^{-3}$"),
        (nong, r"$< n_{\mathrm{e,line}} >/n_G$", ""),
        (tepeak, r"$T_{e0}/ < T_e >$", ""),
        (nepeak, r"$n_{e0}/ < n_{\mathrm{e, vol}} >$", ""),
        ("zeff", r"$Z_{\mathrm{eff}}$", ""),
        (dnz, r"$n_Z/ < n_{\mathrm{e, vol}} >$", ""),
        ("taueff", r"$\tau_e$", "s"),
        ("hfact", "H-factor", ""),
        (pthresh, "H-mode threshold", "MW"),
        ("tauelaw", "Scaling law", ""),
    ]

    plot_info(axis, data, mfile_data, scan)


def plot_magnetics_info(axis, mfile_data, scan):
    """Function to plot magnet info

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    # Load dicts from dicts JSON file
    dicts = get_dicts()

    # Check for Copper magnets
    i_tf_sup = (
        int(mfile_data.data["i_tf_sup"].get_scan(scan))
        if "i_tf_sup" in mfile_data.data
        else 1
    )

    axis.text(-0.05, 1, "Coil currents etc:", ha="left", va="center")
    axis.set_ylim([-16, 1])
    axis.set_xlim([0, 1])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)

    # Number of coils (1 is OH coil)
    number_of_coils = 0
    for item in mfile_data.data:
        if "rpf[" in item:
            number_of_coils += 1

    pf_info = []
    for i in range(1, number_of_coils):
        if i % 2 != 0:
            pf_info.append(
                (
                    mfile_data.data[f"ric[{i:01}]"].get_scan(scan),
                    f"PF {i}",
                )
            )

    if len(pf_info) > 2:
        pf_info_3_a = pf_info[2][0]
        pf_info_3_b = pf_info[2][1]
    else:
        pf_info_3_a = ""
        pf_info_3_b = ""

    tburn = mfile_data.data["tburn"].get_scan(scan) / 3600.0

    i_tf_bucking = (
        int(mfile_data.data["i_tf_bucking"].get_scan(scan))
        if "i_tf_bucking" in mfile_data.data
        else 1
    )

    # Get superconductor material (i_tf_sc_mat)
    # If i_tf_sc_mat not present, assume resistive
    i_tf_sc_mat = (
        int(mfile_data.data["i_tf_sc_mat"].get_scan(scan))
        if "i_tf_sc_mat" in mfile_data.data
        else 0
    )

    tftype = (
        dicts["DICT_TF_TYPE"][str(int(mfile_data.data["i_tf_sc_mat"].get_scan(scan)))]
        if i_tf_sc_mat > 0
        else "Resistive"
    )

    vssoft = mfile_data.data["vsres"].get_scan(scan) + mfile_data.data["vsind"].get_scan(
        scan
    )

    sig_case = 1.0e-6 * mfile_data.data[f"sig_tf_tresca_max({i_tf_bucking})"].get_scan(
        scan
    )
    sig_cond = 1.0e-6 * mfile_data.data[
        f"sig_tf_tresca_max({i_tf_bucking + 1})"
    ].get_scan(scan)
    alstrtf = 1.0e-6 * mfile_data.data["alstrtf"].get_scan(scan)

    data = [
        (pf_info[0][0], pf_info[0][1], "MA"),
        (pf_info[1][0], pf_info[1][1], "MA"),
        (pf_info_3_a, pf_info_3_b, "MA"),
        (vssoft, "Startup flux swing", "Wb"),
        ("vstot", "Available flux swing", "Wb"),
        (tburn, "Burn time", "hrs"),
        ("", "", ""),
        (f"#TF coil type is {tftype}", "", ""),
    ]
    data2 = [
        (sig_cond, "TF Cond max TRESCA stress", "MPa"),
        (sig_case, "TF Case max TRESCA stress", "MPa"),
        (alstrtf, "Allowable stress", "Pa"),
    ]
    if i_tf_sup == 1:
        data += [
            ("bmaxtfrp", "Peak field at conductor (w. rip.)", "T"),
            *data2,
            ("iooic", "I/I$_{\\mathrm{crit}}$", ""),
            ("tmargtf", "TF Temperature margin", "K"),
            ("tmargoh", "CS Temperature margin", "K"),
            ("whttf/n_tf", "Mass per TF coil", "kg"),
        ]
    else:
        n_tf = mfile_data.data["n_tf"].get_scan(scan)
        prescp = 1.0e-6 * mfile_data.data["prescp"].get_scan(scan)
        presleg = 1.0e-6 * mfile_data.data["presleg"].get_scan(scan)
        pres_joints = 1.0e-6 * mfile_data.data["pres_joints"].get_scan(scan)
        fcoolcp = 100.0 * mfile_data.data["fcoolcp"].get_scan(scan)

        data += [
            ("bmaxtf", "Peak field at conductor (w. rip.)", "T"),
            ("ritfc", "TF coil currents sum", "A"),
            ("", "", ""),
            ("#TF coil forces/stresses", "", ""),
            *data2,
            (fcoolcp, "CP cooling fraction", "%"),
            ("vcool", "Maximum coolant flow speed", "m.s$^{-1}$"),
            (prescp, "CP Resisitive heating", "MW"),
            (presleg * n_tf, "legs Resisitive heating (all legs)", "MW"),
            (pres_joints, "TF joints resisitive heating ", "MW"),
        ]

    plot_info(axis, data, mfile_data, scan)


def plot_power_info(
    axis,
    mfile_data,
    scan,
    pgrossmw,
    pthermmw,
    htpmw,
    pnetelmw,
    powfmw,
    crypmw,
    ipedestal,
):
    """Function to plot power info

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    axis.text(-0.05, 1, "Power flows:", ha="left", va="center")
    axis.set_ylim([-16, 1])
    axis.set_xlim([0, 1])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)

    # Define appropriate pedestal and impurity parameters
    if ipedestal == 1:
        ped_height = ("neped", "Electron density at pedestal", "m$^{-3}$")
        ped_pos = ("rhopedn", "r/a at density pedestal", "")
    else:
        ped_height = ("", "No pedestal model used", "")
        ped_pos = ("", "", "")

    data = [
        ("wallmw", "Nominal neutron wall load", "MW m$^{-2}$"),
        ("coreradius", "Normalised radius of 'core' region", ""),
        ped_height,
        ped_pos,
        ("ralpne", "Helium fraction", ""),
        ("pinnerzoneradmw", "inner zone radiation", "MW"),
        ("pradmw", "Total radiation in LCFS", "MW"),
        ("pnucblkt", "Nuclear heating in blanket", "MW"),
        ("pnucshld", "Nuclear heating in shield", "MW"),
        (crypmw, "TF cryogenic power", "MW"),
        ("pdivt", "Power to divertor", "MW"),
        ("divlife", "Divertor life", "years"),
        ("pthermmw", "Primary (high grade) heat", "MW"),
        (100.0 * pgrossmw / pthermmw, "Gross cycle efficiency", "%"),
        (100.0 * (pgrossmw - htpmw) / (pthermmw - htpmw), "Net cycle efficiency", "%"),
        ("pgrossmw", "Gross electric power", "MW"),
        ("pnetelmw", "Net electric power", "MW"),
        (
            100.0 * pnetelmw / powfmw,
            "Fusion-to-electric efficiency "
            + r"$\frac{P_{\mathrm{e,net}}}{P_{\mathrm{fus}}}$",
            "%",
        ),
    ]

    plot_info(axis, data, mfile_data, scan)


class CurrentDrive(Enum):
    NBI = auto()
    ECRH = auto()
    EBW = auto()


def plot_current_drive_info(axis, mfile_data, scan):
    """Function to plot current drive info

    Arguments:
        axis --> axis object to plot to
        mfile_data --> MFILE.DAT object
        scan --> scan number to use

    """
    iefrf = mfile_data.data["iefrf"].get_scan(scan)
    if iefrf in (5, 8):
        drive_type = CurrentDrive.NBI
        title = "Neutral Beam Current Drive:"
    elif iefrf in (3, 7, 10, 11):
        drive_type = CurrentDrive.ECRH
        title = "Electron Cyclotron Current Drive:"
    elif iefrf == 12:
        drive_type = CurrentDrive.EBW
        title = "Electron Bernstein Wave Drive:"
    elif iefrf in (1, 2, 4, 6, 9):
        raise NotImplementedError(
            "Options 1, 2, 4, 6 and 9 not implemented yet in this python script plot_proc.py\n"
            "NEEDS TO BE IMPLEMENTED in plot_current_drive_info subroutine!!\n"
        )
    else:
        raise ValueError(f"Unknown iefrf value: {iefrf}")
    axis.text(-0.05, 1, title, ha="left", va="center")

    if "iefrffix" in mfile_data.data:
        iefrffix = mfile_data.data["iefrffix"].get_scan(scan)

        if iefrffix in (5, 8):
            secondary_heating = "NBI"
        elif iefrffix in (3, 7, 10, 11):
            secondary_heating = "ECH"
        elif iefrffix == 12:
            secondary_heating = "EBW"
        elif iefrffix in (1, 2, 4, 6, 9):
            raise NotImplementedError(
                "Options 1, 2, 4, 6 and 9 not implemented yet in this python script plot_proc.py\n"
                "NEEDS TO BE IMPLEMENTED in plot_current_drive_info subroutine!!\n"
            )
        else:
            raise ValueError(f"Unknown iefrffix value {iefrffix}")

    axis.set_ylim([-16, 1])
    axis.set_xlim([0, 1])
    axis.set_axis_off()
    axis.set_autoscaley_on(False)
    axis.set_autoscalex_on(False)

    rmajor = mfile_data.data["rmajor"].get_scan(scan)
    pinjie = mfile_data.data["pinjmw"].get_scan(scan)
    pinjmwfix = mfile_data.data["pinjmwfix"].get_scan(scan)
    pdivt = mfile_data.data["pdivt"].get_scan(scan)
    pdivr = pdivt / rmajor

    pdivnr = 1.0e20 * pdivr / mfile_data.data["dene"].get_scan(scan)

    # Assume Martin scaling if pthresh is not printed
    # Accounts for pthresh not being written prior to issue #679 and #680
    pthresh = mfile_data.data[
        "plhthresh" if "plhthresh" in mfile_data.data else "pthrmw(6)"
    ].get_scan(scan)
    flh = pdivt / pthresh

    powerht = mfile_data.data["powerht"].get_scan(scan)
    psync = mfile_data.data["psyncpv*vol"].get_scan(scan)
    pbrem = mfile_data.data["pinnerzoneradmw"].get_scan(scan)
    hfact = mfile_data.data["hfact"].get_scan(scan)
    hstar = hfact * (powerht / (powerht + psync + pbrem)) ** 0.31

    data = [
        (pinjie, "Steady state auxiliary power", "MW"),
        ("pheat", "Power for heating only", "MW"),
        ("bootipf", "Bootstrap fraction", ""),
        ("faccd", "Auxiliary fraction", ""),
        ("facoh", "Inductive fraction", ""),
    ]

    if drive_type is CurrentDrive.NBI:
        data += [
            ("gamnb", "NB gamma", "$10^{20}$ A W$^{-1}$ m$^{-2}$"),
            ("powerht", "Plasma heating used for H factor", "MW"),
        ]

    data += [
        ("powerht", "Plasma heating used for H factor", "MW"),
        (pdivr, r"$\frac{P_{\mathrm{div}}}{R_{0}}$", "MW m$^{-1}$"),
        (pdivnr, r"$\frac{P_{\mathrm{div}}}{<n> R_{0}}$", r"$\times 10^{-20}$ MW m$^{2}$"),
        (flh, r"$\frac{P_{\mathrm{div}}}{P_{\mathrm{LH}}}$", ""),
        (hstar, "H* (non-rad. corr.)", ""),
    ]

    if "iefrffix" in mfile_data.data:
        data.insert(
            1, ("pinjmwfix", f"{secondary_heating} secondary auxiliary power", "MW")
        )
        data[0] = (pinjie - pinjmwfix, "Primary auxiliary power", "MW")
        data.insert(2, (pinjie, "Total auxillary power", "MW"))

    coe = mfile_data.data["coe"].get_scan(scan)
    data.append(("", "", ""))
    data.append(("#Costs", "", ""))
    data.append(
        ("", "Cost output not selected", "")
        if coe == 0.0
        else (coe, "Cost of electricity", r"\$/MWh")
    )

    plot_info(axis, data, mfile_data, scan)


def main_plot(
    fig1,
    fig2,
    plot_data,
    colour_dict,
    m_file_data,
    scan,
    imp: Union[Path, str] = "../data/lz_non_corona_14_elements/",
    demo_ranges=False,
):
    """Function to create radial and vertical build plot on given figure.

    Arguments:
      fig1 --> figure object to add plot to.
      fig2 --> figure object to add plot to.
      m_file_data --> MFILE.DAT data to read
      scan --> scan to read from MFILE.DAT
      imp --> path to impurity data
    """
    # Checking the impurity data folder
    # Get path to impurity data dir
    # TODO use Path objects throughout module, not strings
    with resources.path(
        "process.data.lz_non_corona_14_elements", "Ar_lz_tau.dat"
    ) as imp_path:
        data_folder = imp_path.parent

    imp = Path(imp)
    if data_folder.is_dir():
        imp = data_folder
    elif not imp.is_dir():
        print(
            "\033[91m Warning : Impossible to recover impurity data,"
            " try running the macro in the main/utility folder\n"
            "          -> No impurity plot done\033[0m"
        )

    # Plot poloidal cross-section
    plot_1 = fig2.add_subplot(221, aspect="equal")
    poloidal_cross_section(plot_1, m_file_data, plot_data, scan, demo_ranges)

    # Plot toroidal cross-section
    plot_2 = fig2.add_subplot(222, aspect="equal")
    toroidal_cross_section(
        plot_2, m_file_data, plot_data, colour_dict, scan, demo_ranges
    )

    # Plot color key
    plot_3 = fig2.add_subplot(241)
    color_key(plot_3)

    # Plot density profiles
    plot_4 = fig2.add_subplot(234)  # , aspect= 0.05)
    plot_nprofile(plot_4, plot_data, demo_ranges)

    # Plot temperature profiles
    plot_5 = fig2.add_subplot(235)  # , aspect= 1/35)
    plot_tprofile(plot_5, plot_data, demo_ranges)

    if imp.is_dir():
        # plot_qprofile(plot_6)
        plot_6 = fig2.add_subplot(236)  # , aspect=2)
        plot_radprofile(plot_6, m_file_data, plot_data, scan, imp, demo_ranges)

    # Setup params for text plots
    plt.rcParams.update({"font.size": 8})

    # Plot header info
    plot_1 = fig1.add_subplot(231)
    plot_header(plot_1, m_file_data, scan)

    # Geometry
    plot_2 = fig1.add_subplot(232)
    plot_geometry_info(plot_2, m_file_data, plot_data, scan)

    # Physics
    plot_3 = fig1.add_subplot(233)
    plot_physics_info(plot_3, m_file_data, scan)

    # Magnetics
    plot_4 = fig1.add_subplot(234)
    plot_magnetics_info(plot_4, m_file_data, scan)

    # power/flow economics
    plot_5 = fig1.add_subplot(235)
    plot_power_info(plot_5, m_file_data, scan, **plot_data.power_info())

    # Current drive
    plot_6 = fig1.add_subplot(236)
    plot_current_drive_info(plot_6, m_file_data, scan)
    fig1.subplots_adjust(wspace=0.25)


def _create_fig_save(figno, aspect: Union[Literal["equal", "auto"], float] = "equal"):
    fig = plt.figure(figsize=(12, 9), dpi=80)
    return fig, fig.add_subplot(figno, aspect=aspect)


def save_plots(m_file_data, plot_data, colour_dict, demo_ranges, scan):
    """Function to recreate and save individual plots."""
    # Plot poloidal cross-section
    fig, pol = _create_fig_save(111)
    poloidal_cross_section(pol, m_file_data, plot_data, scan, False)

    # Plot TF coils
    plot_tf_coils(pol, m_file_data, plot_data, scan)

    # Plot PF coils
    plot_pf_coils(pol, m_file_data, scan)

    fig.savefig("psection.svg", format="svg", dpi=1200)

    # Plot toroidal cross-section
    fig, tor = _create_fig_save(222)
    toroidal_cross_section(tor, m_file_data, plot_data, colour_dict, scan, False)
    fig.savefig("tsection.svg", format="svg", dpi=1200)

    # Plot color key
    fig, plot = _create_fig_save(241, aspect="auto")
    color_key(plot)
    fig.savefig("color_key.svg", format="svg", dpi=1200)

    # Plot profiles
    fig, plot = _create_fig_save(223, aspect=0.05)
    plot_nprofile(plot, False, plot_data)
    fig.savefig("nprofile.svg", format="svg", dpi=1200)

    fig, plot = _create_fig_save(224, aspect=1 / 35)
    plot_tprofile(plot, False, plot_data)
    fig.savefig("tprofile.svg", format="svg", dpi=1200)


def test(f):
    """Test Function

    :param f: filename to test
    """
    try:
        with patch("bpdf.PdfPages"):
            main([f"-n=-1 -f={f}"])
        return True
    except Exception:
        print(f"FTest failure for file : {f}")
        return False


def parse_args(args):
    """Parse supplied arguments.

    :param args: arguments to parse
    :type args: list, None
    :return: parsed arguments
    :rtype: Namespace
    """
    # Setup command line arguments
    parser = argparse.ArgumentParser(
        description="Produces a two page summary of the PROCESS MFILE output, using the MFILE.  "
        "For info contact michael.kovari@ukaea.uk or james.morris2@ukaea.uk.  "
    )

    parser.add_argument(
        "-f",
        metavar="FILENAME",
        type=str,
        default="",
        help="specify input/output file path",
    )

    parser.add_argument("-s", "--show", help="show plot", action="store_true")

    parser.add_argument("-n", type=int, help="Which scan to plot?")

    parser.add_argument(
        "-d",
        "--DEMO_ranges",
        help="Uses the DEMO dimensions as ranges for all graphics",
        action="store_true",
    )

    parser.add_argument(
        "--svg",
        help="Save plots individually as svg",
        action="store_true",
    )
    return parser.parse_args(args)


def main(args=None):
    # working with minimal changes. Should be converted to class structure
    args = parse_args(args)

    # read MFILE
    m_file = mf.MFile(args.f) if args.f != "" else mf.MFile("MFILE.DAT")

    scan = args.n if args.n else -1

    demo_ranges = args.DEMO_ranges

    plot_data = PlotData.from_mfile(m_file, scan)

    colour_dict = {
        "ohcth": solenoid,
        "precomp": cscompression,
        "tfcth": tfc,
        "thshield_ib": thermal_shield,
        "thshield_ob": thermal_shield,
        "thshield_vb": thermal_shield,
        "d_vv_in": vessel,
        "d_vv_out": vessel,
        "d_vv_top": vessel,
        "d_vv_bot": vessel,
        "shldith": shield,
        "shldoth": shield,
        "blnkith": blanket,
        "blnkoth": blanket,
        "fwith": firstwall,
        "fwoth": firstwall,
        "rminor": plasma,
    }

    # read MFILE
    # m_file = mf.MFile(args.f)
    # scan = scan

    # create main plot
    page1 = plt.figure(figsize=(12, 9), dpi=80)
    page2 = plt.figure(figsize=(12, 9), dpi=80)

    # run main_plot
    main_plot(
        page1, page2, plot_data, colour_dict, m_file, scan=scan, demo_ranges=demo_ranges
    )

    with bpdf.PdfPages(args.f + "SUMMARY.pdf") as pdf:
        pdf.savefig(page1)
        pdf.savefig(page2)

    # show fig if option used
    if args.show:
        plt.show(block=True)

    if args.svg:
        save_plots(m_file, plot_data, colour_dict, demo_ranges, scan)
    plt.close(page1)
    plt.close(page2)


if __name__ == "__main__":
    main()
