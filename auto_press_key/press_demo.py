"""rover2026 arm typing a code on the keypad — Isaac Sim 6.0.1 GUI demo.

Kinematic: IK + forward kinematics from the URDF; each link's USD transform is written
every frame (no physics stepping), so what you see is exactly the planned motion.
"""
import os
import math
import xml.etree.ElementTree as ET

import isaacsim
from isaacsim import SimulationApp

KIT = os.path.join(os.path.dirname(isaacsim.__file__), "apps", "isaacsim.exp.full.kit")
app = SimulationApp(
    {"headless": False, "width": 1920, "height": 1200, "window_width": 1920, "window_height": 1170},
    experience=KIT,
)

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
ARM_USD = f"{ASSETS}/ee_usd/rover2026_ee/rover2026_ee.usda"  # arm + temp presser EE
KEYPAD_USD = f"{ASSETS}/keypad.usd"
URDF = f"{ASSETS}/rover2026_ee.urdf"

LAYOUT = {  # keyboard_stack_v2/layout.py (RoboSuite frame)
    0: (0.11, -0.04, 0.828), 1: (0.05, -0.08, 0.828), 2: (0.07, -0.08, 0.828), 3: (0.09, -0.08, 0.828),
    4: (0.05, -0.06, 0.828), 5: (0.07, -0.06, 0.828), 6: (0.09, -0.06, 0.828), 7: (0.05, -0.04, 0.828),
    8: (0.07, -0.04, 0.828), 9: (0.09, -0.04, 0.828),
}
# Put the keypad on a table in front of the arm (layout.py coords assume RoboSuite's robot mount).
KP_OFFSET = np.array([0.0, -0.49, -0.528])
KEY_PTS = {k: np.array(v) + KP_OFFSET for k, v in LAYOUT.items()}
KEY_WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine"}
CODE = [4, 8, 1, 5, 9, 0]
HOVER = 0.06    # m above key top
PRESS = -0.002  # m below key top at full press


def log(*a):
    print("[demo]", *a, flush=True)


# ---------------------------------------------------------------- URDF kinematics
def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def ax_R(a, t):
    a = np.asarray(a, float) / (np.linalg.norm(a) + 1e-12)
    x, y, z = a
    c, s = math.cos(t), math.sin(t)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def H(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


root = ET.parse(URDF).getroot()
JOINTS = []
for j in root.findall("joint"):
    if j.get("type") not in ("revolute", "continuous"):
        continue
    o = j.find("origin")
    lim = j.find("limit")
    lo, hi = float(lim.get("lower", -math.pi)), float(lim.get("upper", math.pi))
    if j.get("type") == "continuous" or hi <= lo:
        lo, hi = -math.pi, math.pi
    JOINTS.append({
        "name": j.get("name"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
        "xyz": np.array([float(v) for v in o.get("xyz").split()]),
        "rpy": np.array([float(v) for v in o.get("rpy").split()]),
        "axis": np.array([float(v) for v in j.find("axis").get("xyz").split()]),
        "lo": lo, "hi": hi,
    })
BOUNDS = [(j["lo"], j["hi"]) for j in JOINTS]
FIXED = []  # presser mount + tip frame
for j in root.findall("joint"):
    if j.get("type") == "fixed":
        o = j.find("origin")
        FIXED.append((j.find("parent").get("link"), j.find("child").get("link"),
                      [float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]))


def fk_frames(q):
    frames = {"base_link": np.eye(4)}
    for J, qi in zip(JOINTS, q):
        Tj = frames[J["parent"]] @ H(rpy_R(*J["rpy"]), J["xyz"])
        frames[J["child"]] = Tj @ H(ax_R(J["axis"], qi), np.zeros(3))
    for par, ch, xyz, rpy in FIXED:
        if par in frames:
            frames[ch] = frames[par] @ H(rpy_R(*rpy), np.array(xyz))
    return frames


# Tool direction + tip distance: straight from the holder origin to the presser tip frame.
_F0 = fk_frames(np.zeros(len(JOINTS)))
_tip_local = np.linalg.inv(_F0["a6_EE_holder"]) @ _F0["ee_tip"]
TIP = float(np.linalg.norm(_tip_local[:3, 3]))
TOOL_DIR = _tip_local[:3, 3] / TIP  # in a6_EE_holder (== ee_tip) frame
log(f"presser tip {TIP*1000:.0f} mm from EE-holder origin along {np.round(TOOL_DIR, 2)} (link frame)")


def fk(q):
    T = fk_frames(q)["ee_tip"]
    return T[:3, 3], T[:3, :3] @ TOOL_DIR


DOWN = np.array([0.0, 0.0, -1.0])


def ik(target, seed, w_axis=1.0):
    """target = where the presser TIP must be; presser points straight down."""
    target = np.asarray(target, float)

    def cost(q):
        p, tool = fk(q)
        return (1e4 * np.sum((p - target) ** 2) + w_axis * (1.0 - float(np.dot(tool, DOWN)))
                + 1e-3 * np.sum((q - seed) ** 2))

    r = minimize(cost, seed, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 400})
    return r.x, float(np.linalg.norm(fk(r.x)[0] - target))


def solve_all():
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    up = np.zeros(3)  # FK/IK track the presser tip itself
    centre = np.mean([KEY_PTS[k] for k in CODE], axis=0) + up + np.array([0, 0, HOVER + 0.05])
    best = None
    for _ in range(60):
        q, e = ik(centre, lo + (hi - lo) * rng.random(len(lo)))
        score = e + 0.05 * (1 - float(np.dot(fk(q)[1], DOWN)))
        if best is None or score < best[2]:
            best = (q, e, score)
    log(f"ready pose err {best[1]*1000:.1f} mm, tool·down {np.dot(fk(best[0])[1], DOWN):.3f}")
    plan, seed = {}, best[0]
    for k in CODE:
        top = KEY_PTS[k] + up
        qh, eh = ik(top + [0, 0, HOVER], seed)
        qp, ep = ik(top + [0, 0, PRESS], qh)
        plan[k] = (qh, qp)
        seed = qh
        log(f"key {k}: hover err {eh*1000:.1f} mm, press err {ep*1000:.1f} mm, "
            f"tool·down {np.dot(fk(qp)[1], DOWN):.3f}")
    return best[0], plan


READY, PLAN = solve_all()

# ---------------------------------------------------------------- scene
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

add_reference_to_stage(ARM_USD, "/World/rover")
add_reference_to_stage(KEYPAD_USD, "/World/Keypad")


def mat(path, rgb, metallic=0.1, rough=0.45, emissive=None, emissive_k=0.0):
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    if emissive:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[c * emissive_k for c in emissive]))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def bind(prim, m):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.strongerThanDescendants)


M = {
    "base": mat("/World/Looks/base", (0.13, 0.14, 0.17), 0.6, 0.35),
    "blue": mat("/World/Looks/blue", (0.05, 0.25, 0.75), 0.5, 0.3),
    "blue2": mat("/World/Looks/blue2", (0.10, 0.45, 0.90), 0.5, 0.3),
    "teal": mat("/World/Looks/teal", (0.02, 0.60, 0.62), 0.4, 0.3),
    "ee": mat("/World/Looks/ee", (0.98, 0.42, 0.05), 0.3, 0.35),
    "presser": mat("/World/Looks/presser", (0.92, 0.92, 0.95), 0.9, 0.2),
    "plate": mat("/World/Looks/plate", (0.05, 0.06, 0.08), 0.2, 0.6),
    "stand": mat("/World/Looks/stand", (0.22, 0.22, 0.25), 0.7, 0.4),
    "key": mat("/World/Looks/key", (0.86, 0.87, 0.90), 0.0, 0.5),
    "target": mat("/World/Looks/target", (1.00, 0.80, 0.05), 0.0, 0.4, (1.0, 0.75, 0.05), 3.0),
    "pressed": mat("/World/Looks/pressed", (0.10, 0.95, 0.30), 0.0, 0.4, (0.10, 1.0, 0.30), 8.0),
    "done": mat("/World/Looks/done", (0.35, 0.80, 0.45), 0.0, 0.5),
    "floor": mat("/World/Looks/floor", (0.26, 0.30, 0.36), 0.0, 0.85),
}

LINK_COLOR = {"base_link": "base", "a1_shoulder_base": "blue", "a2_link_1": "blue2", "a3_axis4_housing": "blue",
              "a4_link2": "blue2", "a5_internal_upright": "teal", "a6_EE_holder": "ee", "ee_presser": "presser"}
# importer 6.x nests links (child link prim under parent link prim); first match in pre-order is the link,
# same-named children are its visual meshes.
changed = True  # importer 6.x instances the meshes; instance proxies ignore material bindings
while changed:
    changed = False
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(Sdf.Path("/World/rover")) and prim.IsInstance():
            prim.SetInstanceable(False)
            changed = True
link_prims = {}
for prim in stage.Traverse():
    if prim.GetPath().HasPrefix(Sdf.Path("/World/rover")) and prim.GetName() in LINK_COLOR \
            and prim.GetName() not in link_prims:
        link_prims[prim.GetName()] = prim
for name, prim in link_prims.items():
    for c in prim.GetChildren():  # colour the link's own meshes, not its child links
        if c not in link_prims.values():
            bind(c, M[LINK_COLOR[name]])
log("links:", sorted(link_prims))
if os.environ.get("DEMO_DEBUG"):
    for p in Usd.PrimRange(link_prims["a6_EE_holder"], Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Gprim):
            mb, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            log("DBG", str(p.GetPath())[-50:], p.GetTypeName(), UsdGeom.Imageable(p).ComputePurpose(),
                UsdGeom.Imageable(p).ComputeVisibility(), "proxy" if p.IsInstanceProxy() else "",
                mb.GetPath() if mb else None, p.IsActive())

# every moving link gets one local matrix op (joint origin * joint rotation) driven each frame
link_ops = {}
for J in JOINTS:
    xf = UsdGeom.Xformable(link_prims[J["child"]])
    xf.ClearXformOpOrder()
    link_ops[J["child"]] = xf.AddTransformOp()

key_prims = {}
for prim in stage.Traverse():
    if not prim.GetPath().HasPrefix(Sdf.Path("/World/Keypad")):
        continue
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    n = prim.GetName()
    if n == "plate":
        bind(prim, M["plate"])
    elif n.startswith("key_"):
        key_prims[n[4:]] = prim
        bind(prim, M["key"])
key_rest = {w: Gf.Vec3d(p.GetAttribute("xformOp:translate").Get()) for w, p in key_prims.items()}

# move the keypad onto a small table in front of the arm
UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Keypad")).SetTranslate(Gf.Vec3d(*KP_OFFSET.tolist()))
table_top = 0.816 + KP_OFFSET[2]  # keypad plate bottom (keypad.usd plate centre z=0.82, 8 mm thick)
table = UsdGeom.Cube.Define(stage, "/World/KeypadTable")
table.CreateSizeAttr(1.0)
UsdGeom.XformCommonAPI(table).SetTranslate(Gf.Vec3d(0.08, -0.06 + KP_OFFSET[1], table_top / 2))
UsdGeom.XformCommonAPI(table).SetScale(Gf.Vec3f(0.26, 0.22, table_top))
bind(table.GetPrim(), M["stand"])

floor = UsdGeom.Cube.Define(stage, "/World/Floor")
floor.CreateSizeAttr(1.0)
UsdGeom.XformCommonAPI(floor).SetTranslate(Gf.Vec3d(0, 0, -0.006))
UsdGeom.XformCommonAPI(floor).SetScale(Gf.Vec3f(6, 6, 0.01))
bind(floor.GetPrim(), M["floor"])

dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
dome.CreateIntensityAttr(700)
dome.CreateColorAttr(Gf.Vec3f(0.70, 0.82, 1.0))
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(3200)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.92, 0.80))
sun.CreateAngleAttr(1.0)
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-45, 20, -35))


def pose(q):
    for J, qi in zip(JOINTS, q):
        local = H(rpy_R(*J["rpy"]), J["xyz"]) @ H(ax_R(J["axis"], qi), np.zeros(3))
        link_ops[J["child"]].Set(Gf.Matrix4d(local.T.tolist()))


def set_key(word, state, dz=0.0):
    p = key_prims.get(word)
    if p is None:
        return
    bind(p, M[state])
    r = key_rest[word]
    p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(r[0], r[1], r[2] + dz))


def smooth(a, b, n):
    for i in range(1, n + 1):
        s = i / n
        s = s * s * (3 - 2 * s)
        yield a + (b - a) * s


def run(qs):
    for q in qs:
        if not app.is_running():
            return False
        pose(q)
        app.update()
    return True


pose(READY)
for _ in range(10):
    app.update()
set_camera_view(eye=[0.62, -1.02, 0.60], target=[0.07, -0.50, 0.36], camera_prim_path="/OmniverseKit_Persp")
log("typing code", "".join(map(str, CODE)), "- looping")

cur, cycle = READY, 0
while app.is_running():
    cycle += 1
    for w in key_prims:
        set_key(w, "key")
    ok = True
    for k in CODE:
        w = KEY_WORD[k]
        qh, qp = PLAN[k]
        set_key(w, "target")
        ok = run(smooth(cur, qh, 70)) and run(smooth(qh, qp, 24))
        if not ok:
            break
        set_key(w, "pressed", dz=-0.003)
        ok = run([qp] * 16)
        set_key(w, "done")
        ok = ok and run(smooth(qp, qh, 20))
        if not ok:
            break
        cur = qh
    if not ok or not run(smooth(cur, READY, 60)):
        break
    cur = READY
    run([READY] * 60)
    log(f"cycle {cycle} done")

app.close()
