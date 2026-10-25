"""Explicit SE(2) / pinhole conventions. No learned or GT-derived orientation offsets."""
import numpy as np
F_W=1/(2*np.tan(0.698132))
COLUMN_ANGLES=np.flip(np.arctan2(np.arange(40)-19.5,40*F_W)) # image left to right

def axial_to_radial(depth): return np.asarray(depth)/np.cos(COLUMN_ANGLES)
def radial_to_axial(depth): return np.asarray(depth)*np.cos(COLUMN_ANGLES)
def wrap_angle(x): return (np.asarray(x)+np.pi)%(2*np.pi)-np.pi

def pose_error(pose,gt):
    return float(np.linalg.norm(np.asarray(pose)[:2]-gt[:2])),float(abs(wrap_angle(pose[2]-gt[2]))*180/np.pi)

def world_to_image(points,origin,meters_per_pixel):
    p=np.asarray(points)-np.asarray(origin)
    return p*np.array([1.,-1.])/meters_per_pixel

def image_to_world(points,origin,meters_per_pixel):
    return np.asarray(points)*meters_per_pixel*np.array([1.,-1.])+np.asarray(origin)

def zind_crop_bearing(rotation_deg,yaw_deg,global_rotation_deg=0):
    """Map stores rotated ZInD XY as image XY. Positive ray bearing uses -image Y.
    Independent derivation: ZInD theta=atan2(-X,Y), crop center theta=yaw-180.
    """
    return float(wrap_angle(np.deg2rad(90-rotation_deg-global_rotation_deg-yaw_deg)))

def semantic_rays_nearest(labels):
    # Actual pinhole column coordinates for target angles -40..+40deg.
    x=np.clip(np.tan(np.deg2rad(np.arange(-40,41,10)))*40*F_W+19.5,0,39)
    return np.asarray(labels)[np.rint(x).astype(int)]
