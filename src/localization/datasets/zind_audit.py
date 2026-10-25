"""Independent annotation/projection audit; never used to select a query pose online."""
from pathlib import Path
import json,cv2
import numpy as np
from common import ROOT
from zind_utils import pano2persp,PolygonType
from data_utils.zind.create_casting_files_zind import extract_polygons_from_zind,compute_camera_poses_zind,shift_polygons_and_cameras
from data_utils.zind.map_zind_rooms import map_label
from modules.semantic.semantic_mapper import zind_room_type_to_id
from localization.geometry import zind_crop_bearing,wrap_angle

def audit_annotations(manifest):
    source=ROOT/'repos/zind/sample_tour/000';raw=json.loads((source/'zind_data.json').read_text());floor='floor_01'
    global_rot=-raw['floorplan_to_redraw_transformation'][floor]['rotation']
    nodes=[n for r in raw['merger'][floor].values() for partial in r.values() for n in partial.values()]
    by_path={str(source/n['image_path']):n for n in nodes}
    poly=extract_polygons_from_zind(raw,floor,global_rot);cams=compute_camera_poses_zind(raw,floor,global_rot)
    shifted,cams=shift_polygons_and_cameras(poly,cams)
    h,w=cv2.imread(json.loads((Path(manifest[0])/'sample.json').read_text())['map']).shape[:2]
    masks={i:np.zeros((h,w),np.uint8) for i in range(14)};bboxes={i:np.zeros((h,w),np.uint8) for i in range(14)}
    for p in shifted:
        if p.type==PolygonType.PARTIAL_ROOM:
            category=zind_room_type_to_id[map_label(p.name)];pts=np.array(p.points)*100
            cv2.fillPoly(masks[category],[np.round(pts).astype(int)],1)
            x0,y0=pts.min(0);x1,y1=pts.max(0);cv2.rectangle(bboxes[category],(int(x0),int(y0)),(int(np.ceil(x1)),int(np.ceil(y1))),1,-1)
    details=[]
    for path in manifest:
        meta=json.loads((Path(path)/'sample.json').read_text());node=by_path[meta['pano_path']];rotation=node['floor_plan_transformation']['rotation'];yaw=meta['yaw_crop_deg']
        # Independent ZInD source spherical convention: theta=atan2(-X,Y).
        # Crop center equirectangular theta=yaw-180 degrees.
        sphere_theta=np.deg2rad(yaw-180);direction_local=np.array([-np.sin(sphere_theta),np.cos(sphere_theta)])
        angle=np.deg2rad(rotation+global_rot);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        direction_map=R@direction_local;independent=np.arctan2(-direction_map[1],direction_map[0])
        direct=zind_crop_bearing(rotation,yaw,global_rot);angle_error=float(abs(wrap_angle(independent-meta['pose'][2])))
        pano=cv2.imread(meta['pano_path']);crop=pano2persp(pano,80,yaw,0,0,(360,640));loaded=cv2.imread(str(Path(path)/'rgb.png'))
        # Annotation coordinates use camera height 1; transform to metric.
        height=node['camera_height']*node['floor_plan_transformation']['scale']*raw['scale_meters_per_coordinate'][floor]
        details.append({'sample':path,'annotation_image':node['image_path'],'independent_bearing_error_rad':angle_error,'formula_error_rad':float(abs(wrap_angle(direct-independent))),'crop_max_uint8_diff':int(np.abs(crop.astype(int)-loaded.astype(int)).max()),'camera_height_m':height,'roll_pitch':'not supplied per query; official upright panorama assumption, not independently measured','room_label_from_annotation':map_label(node.get('label','undefined'))})
    return details,masks,bboxes
