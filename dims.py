"""
All geometry for the SM-465 transmission model, in millimetres.

Every number below is copied from transmission.scad (the professor's model),
and the part positions use the same formulas as the .scad modules, so you can
check this file line by line against transmission.scad.

NOTE: the dimension labels in the PDF figures (Figs 5-7) are drawn at 2x scale
(e.g. "660mm" shaft = 330 in the .scad, "160mm" bore = 2 * 40 radius).
The .scad and the PDF text ([280, 210, 300]) are the real values.

World frame = the OpenSCAD frame of main_transmission.scad
----------------------------------------------------------
origin : centre of the case's base plate
x      : along the shaft axes. Negative x = front (input), positive x = rear (output)
y      : across the case width (y = 0 is the centre plane)
z      : up. The base plate spans z in [-12.5, 12.5]; the bench top is z = -12.5.

Shaft frame (for each shaft)
----------------------------
origin at one end of the shaft, on its axis; local +x runs along the shaft.
(The .scad builds shafts along local z; ours run along local x.)
"""

# =============================================================================
# Parameters, copied from transmission.scad
# =============================================================================
GEAR_THICKNESS = 25
COLLAR_THICKNESS = 30
SYNCRO_THICKNESS = 8

SYNCRO12_RADIUS = 48
SYNCRO34_RADIUS = 45

GEAR1_RADIUS = 65
GEAR2_RADIUS = 53
GEAR3_RADIUS = 45
GEAR_REV_PRIMARY = 60

COLLAR12_RADIUS = 60
COLLAR34_RADIUS = 60

BEARING_RADIUS = 40
BEARING_OFFSET_HEIGHT = 215
CS_BEARING_OFFSET_HEIGHT = 100

CASE_THICKNESS = 25
CASE_LENGTH = 280       # inside length between the end walls
CASE_WIDTH = 210        # outside width (side walls are inside this)
CASE_HEIGHT = 300

PRIM_SEC_SHAFT_OFFSET = BEARING_OFFSET_HEIGHT - CS_BEARING_OFFSET_HEIGHT   # 115

GEAR1_SEC_RADIUS = PRIM_SEC_SHAFT_OFFSET - GEAR1_RADIUS                    # 50
GEAR2_SEC_RADIUS = PRIM_SEC_SHAFT_OFFSET - GEAR2_RADIUS                    # 62
GEAR3_SEC_RADIUS = PRIM_SEC_SHAFT_OFFSET - GEAR3_RADIUS                    # 70
GEAR_REV_SECONDARY = PRIM_SEC_SHAFT_OFFSET - GEAR_REV_PRIMARY - 15         # 40
COUNTERGEAR_RADIUS = 70

PRIMARY_SHAFT_LENGTH = 330
SECONDARY_SHAFT_LENGTH = CASE_LENGTH + 2 * CASE_THICKNESS                  # 330
SHAFT_RADIUS = 18

# =============================================================================
# Shaft profiles: lists of (t_start, t_end, radius) along the shaft,
# built with the same offsets as primary_shaft_assembly() and
# secondary_shaft_assembly() in transmission.scad.
# Gears and syncros are bounded by their outer (tooth-tip) radius.
# =============================================================================


def _collar(t, radius):
    """collar(): cone 0.9r->r, waist 0.8r, cone r->0.9r. Cones bounded by r."""
    a, b = 0.35 * COLLAR_THICKNESS, 0.65 * COLLAR_THICKNESS
    return [(t, t + a, radius), (t + a, t + b, 0.8 * radius), (t + b, t + COLLAR_THICKNESS, radius)]


def _mainshaft_profile():
    s, c, g = SYNCRO_THICKNESS, COLLAR_THICKNESS, GEAR_THICKNESS
    o = 20  # shaft_offset
    p = [(0, PRIMARY_SHAFT_LENGTH, SHAFT_RADIUS)]                       # shaft
    p += [(o, o + s, SYNCRO34_RADIUS)]                                  # 3-4 syncro
    p += _collar(o + s, COLLAR34_RADIUS)                                # 3-4 collar
    p += [(o + s + c, o + 2*s + c, SYNCRO34_RADIUS)]                    # 3-4 syncro
    p += [(o + 2*s + c, o + 2*s + c + g, GEAR3_RADIUS)]                 # 3rd gear
    p += [(o + 2*s + c + g, o + 2*s + c + 2*g, GEAR2_RADIUS)]           # 2nd gear
    p += [(o + 2*s + c + 2*g, o + 3*s + c + 2*g, SYNCRO12_RADIUS)]      # 1-2 syncro
    p += [(o + 3*s + c + 2*g, o + 3*s + c + 3*g, GEAR_REV_PRIMARY)]     # reverse driven
    p += _collar(o + 3*s + c + 3*g, COLLAR12_RADIUS)                    # 1-2 collar
    p += [(o + 3*s + 2*c + 3*g, o + 4*s + 2*c + 3*g, SYNCRO12_RADIUS)]  # 1-2 syncro
    p += [(o + 4*s + 2*c + 3*g, o + 4*s + 2*c + 4*g, GEAR1_RADIUS)]     # 1st gear
    return p


def _countershaft_profile():
    s, c, g = SYNCRO_THICKNESS, COLLAR_THICKNESS, GEAR_THICKNESS
    o = 38  # shaft_offset
    L = SECONDARY_SHAFT_LENGTH
    p = [(0, L, SHAFT_RADIUS)]                                          # shaft
    p += [(o, o + g, GEAR1_SEC_RADIUS)]                                 # first gear
    p += [(o + s + c + g, o + s + c + 2*g, GEAR_REV_SECONDARY)]         # reverse
    p += [(o + 2*s + c + 2*g, o + 2*s + c + 3*g, GEAR2_SEC_RADIUS)]     # second gear
    p += [(o + 2*s + c + 3*g, o + 2*s + c + 4*g, GEAR3_SEC_RADIUS)]     # third gear
    t = L - 2 * CASE_THICKNESS - 5
    p += [(t, t + g, COUNTERGEAR_RADIUS)]                               # counter gear
    return p


MAINSHAFT_PROFILE = _mainshaft_profile()
COUNTERSHAFT_PROFILE = _countershaft_profile()

# =============================================================================
# Placement in the world, from module transmission()
# =============================================================================
# Mainshaft: translate([-85, 0, 215 + 12.5]) rotate([0, 90, 0])  -> runs toward +x
MAINSHAFT_START_XYZ = (-85.0, 0.0, BEARING_OFFSET_HEIGHT + 0.5 * CASE_THICKNESS)      # z = 227.5

# Countershaft: translate([140 + 25, 0, 100 + 12.5]) rotate([0, 90, 180]) -> runs toward -x
COUNTERSHAFT_ORIGIN_XYZ = (0.5 * CASE_LENGTH + CASE_THICKNESS, 0.0,
                           CS_BEARING_OFFSET_HEIGHT + 0.5 * CASE_THICKNESS)          # (165, 0, 112.5)

# Mainshaft pose shown in apart_transmission.scad (lifted 120, tilted 25 deg).
# Handy as a sanity check: a known pose partway out of the case.
APART_XYZ = (-100.0, 0.0, 120 + BEARING_OFFSET_HEIGHT + 0.5 * CASE_THICKNESS)       # z = 347.5
APART_TILT_DEG = 25.0   # .scad rotate([0,115,0]) on a z-axis shaft = 25 deg about y for our x-axis shaft

# In this idealised model the assembled pose is ON the contact boundary:
#  * the three meshing gear pairs are exactly tangent
#    (65+50, 53+62, 45+70 = 115 = shaft spacing), and
#  * main 2nd gear (r53, x 6..31) and countershaft 3rd gear (r70, x -19..6)
#    sit face-to-face at x = 6 with 8 mm of radial overlap.
# Shrinking every gear/collar/syncro by these amounts (radius, and each flat
# face) gives the start pose a little clearance. Mention this in your write-up.
GEAR_RADIAL_CLEARANCE = 0.5
GEAR_AXIAL_CLEARANCE = 0.5

# =============================================================================
# Workbench / workspace (our additions - not in the .scad)
# =============================================================================
BENCH_TOP_Z = -0.5 * CASE_THICKNESS          # the case's base plate sits on the bench
BENCH_EXTENT = ((-600.0, 700.0), (-600.0, 400.0))   # x, y extent of the bench top
BENCH_THICKNESS = 30.0

# Goal: mainshaft lying on the bench beside the case, axis parallel to x.
GOAL_Y = -250.0

# Sampling bounds for the mainshaft's reference point (its front end), world frame.
WORKSPACE = ((-450.0, 450.0), (-400.0, 250.0), (BENCH_TOP_Z, 600.0))


#dims from importing scad into AI and drawing out measurments to save time
