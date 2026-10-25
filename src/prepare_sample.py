"""Prepare only Zillow's explicitly published one-home sample; no dataset mirrors."""
from common import *
import cv2
from PIL import Image
from data_utils.zind.create_casting_files_zind import extract_polygons_from_zind,compute_camera_poses_zind,shift_polygons_and_cameras,dataset_to_ray_cast
from data_utils.zind.render_floorplans_zind import render_jpg_image
from zind_utils import pano2persp,PolygonType
from utils.raycast_utils import ray_cast

def main():
    src=ROOT/'repos/zind/sample_tour/000'; out=ROOT/'data/zind_public_sample'; out.mkdir(parents=True,exist_ok=True)
    data=json.loads((src/'zind_data.json').read_text()); floor='floor_01'
    rot=-data['floorplan_to_redraw_transformation'][floor]['rotation']
    polys=extract_polygons_from_zind(data,floor,rot)
    cams=compute_camera_poses_zind(data,floor,rot)
    polys,cams=shift_polygons_and_cameras(polys,cams)
    m=render_jpg_image(polys,polygon_list_points=[p.points for p in polys],rendering_type='semantic')
    cv2.imwrite(str(out/'floorplan_semantic.png'),m)
    free=np.zeros(m.shape[:2],np.uint8)
    for p in polys:
        if p.type==PolygonType.ROOM: cv2.fillPoly(free,[np.round(np.array(p.points)*100).astype('int32')],1)
    # OpenCV output is BGR. Convert once for official RGB color classifier.
    rgb=cv2.cvtColor(m,cv2.COLOR_BGR2RGB)/255.
    free[np.any(m<240,axis=2)]=0
    assert free.sum()>1000, 'Empty candidate mask: check PolygonType import identity'
    np.save(out/'free_mask.npy',free)
    angles=np.flip(np.arctan2(np.arange(40)-19.5,40*(1/np.tan(np.deg2rad(40))/2)))
    manifest=[]
    for i in range(min(8,len(cams))):
        x,y,rotation,img_path=cams[i]
        pano=cv2.imread(str(src/img_path))
        for yaw in [0,90,180,270]:
            theta=float((dataset_to_ray_cast(np.deg2rad(rotation))-np.deg2rad(yaw))%(2*np.pi))
            query=pano2persp(pano,80,yaw,0,0,(360,640))
            path=out/'samples'/f'{i:03d}_{yaw:03d}'; path.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(path/'rgb.png'),query)
            hits=[ray_cast(rgb,[x*100,y*100],theta+a,dist_max=1500) for a in angles]
            axial=[float(h[0]*.01*np.cos(a)) for h,a in zip(hits,angles)]
            semantic=[int(h[1]) for h in hits]
            meta={'building_id':'zind_public_sample_000','floor_id':floor,'pano_index':i,'pano_path':str(src/img_path),'yaw_crop_deg':yaw,'pose':[float(x),float(y),theta],'depth_axial_m':axial,'semantic':semantic,'room_label':13,'room_label_available':False,'map':str(out/'floorplan_semantic.png'),'free_mask':str(out/'free_mask.npy'),'map_m_per_pixel':.01,'fov_deg':80,'source':'https://github.com/zillow/zind/tree/main/sample_tour','usage':'engineering_check_only_single_building_not_benchmark'}
            write_json(path/'sample.json',meta); manifest.append(str(path))
    write_json(out/'manifest.json',manifest)
    write_json(out/'split_policy.json',{'train':[],'val':[],'test':[],'engineering_only':manifest,'reason':'Only one public building; no honest cross-building split possible.'})
    print(json.dumps({'samples':len(manifest),'map_shape':m.shape,'manifest':str(out/'manifest.json')}))
if __name__=='__main__': main()
