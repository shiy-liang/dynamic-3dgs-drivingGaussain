import os, glob, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===== paths (repo-relative) =====
IMG_DIR = "data/frames"
OUT_DIR = "outputs/vo"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== load image paths =====
paths = sorted(
    glob.glob(os.path.join(IMG_DIR, "frame_*.jpg")) +
    glob.glob(os.path.join(IMG_DIR, "frame_*.png"))
)

if len(paths) < 10:
    raise RuntimeError(f"Too few frames found in {IMG_DIR}. Found {len(paths)}")

# ===== VO parameters =====
STEP = 2          # IMPORTANT for your 78-frame case
N_FEATURES = 6000
RANSAC_THRESH = 1.5

paths = paths[::STEP]

def prep(img):
    """Resize to stabilize ORB matching"""
    h, w = img.shape
    new_w = 960
    new_h = int(h * new_w / w)
    return cv2.resize(img, (new_w, new_h))

# ===== infer intrinsics from first frame =====
im0 = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
im0 = prep(im0)
H, W = im0.shape

fx = fy = 0.9 * W
cx, cy = W / 2.0, H / 2.0
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

# ===== ORB + matcher =====
orb = cv2.ORB_create(nfeatures=N_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ===== initial pose =====
R = np.eye(3)
t = np.zeros((3, 1))
traj = [t.flatten().copy()]

prev = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
prev = prep(prev)
kp1, des1 = orb.detectAndCompute(prev, None)

good_steps = 0

# ===== main VO loop =====
for i in range(1, len(paths)):
    cur = cv2.imread(paths[i], cv2.IMREAD_GRAYSCALE)
    cur = prep(cur)
    kp2, des2 = orb.detectAndCompute(cur, None)

    if des1 is None or des2 is None or len(kp1) < 50 or len(kp2) < 50:
        kp1, des1 = kp2, des2
        traj.append(t.flatten().copy())
        continue

    matches = bf.match(des1, des2)
    if len(matches) < 80:
        kp1, des1 = kp2, des2
        traj.append(t.flatten().copy())
        continue

    matches = sorted(matches, key=lambda x: x.distance)[:500]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=RANSAC_THRESH
    )

    if E is None:
        kp1, des1 = kp2, des2
        traj.append(t.flatten().copy())
        continue

    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 60:
        kp1, des1 = kp2, des2
        traj.append(t.flatten().copy())
        continue

    _, dR, dt, _ = cv2.recoverPose(E, pts1[inliers], pts2[inliers], K)

    t = t + R @ dt
    R = dR @ R
    traj.append(t.flatten().copy())

    kp1, des1 = kp2, des2
    good_steps += 1

traj = np.array(traj)

# ===== save trajectory =====
csv_path = os.path.join(OUT_DIR, "trajectory.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z"])
    for p in traj:
        writer.writerow([float(p[0]), float(p[1]), float(p[2])])

# ===== plot top-down =====
plt.figure()
plt.plot(traj[:, 0], traj[:, 2])
plt.axis("equal")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("z")
plt.title(f"VO trajectory (top-down), good_steps={good_steps}")
plt.savefig(os.path.join(OUT_DIR, "trajectory.png"), dpi=200)

# ===== plot 3D =====
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("VO trajectory (3D)")
plt.savefig(os.path.join(OUT_DIR, "trajectory3d.png"), dpi=200)

print(f"good_steps: {good_steps} / {len(traj)-1}")
print("Saved to:", OUT_DIR)
