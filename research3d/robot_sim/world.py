"""Generate Gazebo worlds from an existing ProcTHOR house or an original hall.

No external meshes, downloads or location-encoded image markers are used.
The pose publisher is an OFFLINE measurement channel, never an odometry input.
The ProcTHOR path preserves structural geometry, not its furnished appearance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

WORLD_NAME = "airport_hall"
ROBOT_NAME = "robot"
WHEEL_RADIUS = 0.10
WHEEL_SEPARATION = 0.42
WIDTH, HEIGHT = 640, 480
T_BASE_CAMERA_OPTICAL = [[0., 0., 1., .20], [-1., 0., 0., 0.],
                         [0., -1., 0., .50], [0., 0., 0., 1.]]


def _box(name, xyz, size, color, texture=None, collision=True):
    material = f"<ambient>{color} 1</ambient><diffuse>{color} 1</diffuse>"
    if texture:
        material += ("<pbr><metal><albedo_map>" + escape(Path(texture).resolve().as_uri())
                     + "</albedo_map><metalness>0</metalness><roughness>1</roughness></metal></pbr>")
    geom = f"<geometry><box><size>{' '.join(map(str, size))}</size></box></geometry>"
    coll = f'<collision name="collision">{geom}</collision>' if collision else ''
    return f'''<model name="{name}"><static>true</static><pose>{' '.join(map(str, xyz))} 0 0 0</pose>
      <link name="link">{coll}<visual name="visual">{geom}<material>{material}</material></visual></link></model>'''


def robot_sdf(initial_pose=(-2, -1, .101, 0)):
    wheels = []
    for side, y in [('left', .21), ('right', -.21)]:
        wheels.append(f'''<link name="{side}_wheel"><pose>0 {y} 0 -1.5707963267948966 0 0</pose>
          <inertial><mass>0.5</mass><inertia><ixx>.0015</ixx><iyy>.0015</iyy><izz>.0025</izz></inertia></inertial>
          <collision name="collision"><geometry><cylinder><radius>.10</radius><length>.04</length></cylinder></geometry>
            <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
          <visual name="visual"><geometry><cylinder><radius>.10</radius><length>.04</length></cylinder></geometry>
            <material><ambient>.03 .03 .03 1</ambient><diffuse>.05 .05 .05 1</diffuse></material></visual></link>
          <joint name="{side}_wheel_joint" type="revolute"><parent>base_link</parent><child>{side}_wheel</child>
            <axis><xyz>0 0 1</xyz><limit><lower>-1e16</lower><upper>1e16</upper><effort>20</effort><velocity>30</velocity></limit></axis></joint>''')
    px,py,pz,yaw=initial_pose
    return f'''<model name="robot"><pose>{px} {py} {pz} 0 0 {yaw}</pose>
      <link name="base_link"><inertial><pose>0 0 .12 0 0 0</pose><mass>8</mass>
        <inertia><ixx>.18</ixx><iyy>.24</iyy><izz>.30</izz></inertia></inertial>
        <collision name="body"><pose>0 0 .12 0 0 0</pose><geometry><box><size>.45 .32 .18</size></box></geometry></collision>
        <visual name="body"><pose>0 0 .12 0 0 0</pose><geometry><box><size>.45 .32 .18</size></box></geometry>
        <material><ambient>.08 .25 .55 1</ambient><diffuse>.08 .25 .55 1</diffuse></material></visual>
        <collision name="caster"><pose>-.17 0 -.06 0 0 0</pose><geometry><sphere><radius>.04</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <collision name="front_caster"><pose>.17 0 -.06 0 0 0</pose><geometry><sphere><radius>.04</radius></sphere></geometry>
          <surface><friction><ode><mu>0</mu><mu2>0</mu2></ode></friction></surface></collision>
        <sensor name="imu" type="imu"><always_on>true</always_on><update_rate>100</update_rate><topic>/robot/imu</topic>
          <gz_frame_id>base_link</gz_frame_id><imu><angular_velocity>
            <x><noise type="gaussian"><mean>0</mean><stddev>.002</stddev></noise></x>
            <y><noise type="gaussian"><mean>0</mean><stddev>.002</stddev></noise></y>
            <z><noise type="gaussian"><mean>0</mean><stddev>.002</stddev></noise></z>
          </angular_velocity></imu></sensor>
        <sensor name="camera" type="camera"><pose>.20 0 .50 0 0 0</pose><always_on>true</always_on>
          <update_rate>10</update_rate><topic>/robot/camera/image</topic><gz_frame_id>camera_optical_frame</gz_frame_id>
          <camera><horizontal_fov>{math.pi/2}</horizontal_fov><image><width>640</width><height>480</height><format>R8G8B8</format></image>
            <clip><near>.05</near><far>30</far></clip><camera_info_topic>/robot/camera/camera_info</camera_info_topic></camera></sensor>
        <sensor name="camera_right" type="camera"><pose>.20 -.10 .50 0 0 0</pose><always_on>true</always_on>
          <update_rate>10</update_rate><topic>/robot/camera_right/image</topic><gz_frame_id>camera_right_optical_frame</gz_frame_id>
          <camera><horizontal_fov>{math.pi/2}</horizontal_fov><image><width>640</width><height>480</height><format>R8G8B8</format></image>
            <clip><near>.05</near><far>30</far></clip><camera_info_topic>/robot/camera_right/camera_info</camera_info_topic></camera></sensor>
      </link>{''.join(wheels)}
      <plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive">
        <left_joint>left_wheel_joint</left_joint><right_joint>right_wheel_joint</right_joint>
        <wheel_separation>.42</wheel_separation><wheel_radius>.10</wheel_radius><topic>/robot/cmd_vel</topic>
        <odom_topic>/offline/not_for_estimator/diffdrive_odom</odom_topic><tf_topic>/offline/not_for_estimator/diffdrive_tf</tf_topic>
        <max_linear_acceleration>1</max_linear_acceleration><max_angular_acceleration>2</max_angular_acceleration>
        <max_linear_velocity>.7</max_linear_velocity><max_angular_velocity>1</max_angular_velocity>
      </plugin>
      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
        <topic>/robot/joint_states</topic><joint_name>left_wheel_joint</joint_name><joint_name>right_wheel_joint</joint_name>
      </plugin>
      <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
        <publish_link_pose>true</publish_link_pose><publish_model_pose>true</publish_model_pose>
        <publish_nested_model_pose>false</publish_nested_model_pose><publish_sensor_pose>false</publish_sensor_pose>
        <use_pose_vector_msg>true</use_pose_vector_msg><update_frequency>50</update_frequency>
      </plugin></model>'''


def _textures(directory: Path):
    from PIL import Image, ImageDraw
    directory.mkdir(parents=True, exist_ok=True)
    palette = [(25, 71, 120), (31, 108, 81), (150, 81, 36), (92, 63, 122)]
    result = []
    # Labels are ordinary fixed map appearance, independent of camera/query pose.
    for i, label in enumerate(['DEPARTURES', 'ARRIVALS', 'INFORMATION', 'BAGGAGE']):
        im = Image.new('RGB', (512, 384), palette[i]); d = ImageDraw.Draw(im)
        d.rectangle((15, 15, 496, 368), outline='white', width=5)
        for j in range(7):
            x, y = 45 + (j*71+i*19)%410, 70 + (j*43)%220
            d.rectangle((x,y,x+26,y+19), fill=(180+j*8,200-j*10,225))
            d.line((30, 110+j*31, 480, 110+j*31), fill=(110,140,170), width=2)
        d.text((45,30), label, fill='white', font_size=28)
        path = directory / f'poster_{i}.png'; im.save(path); result.append(path)
    return result


def build_world(directory: Path, house_path: Path | None = None):
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    textures = _textures(directory/'textures')
    models, objects = [], []
    def add(name, xyz, size, color, category, texture=None, collision=True):
        models.append(_box(name,xyz,size,color,texture,collision))
        objects.append(dict(name=name,center_m=xyz,size_m=size,category=category,collision=collision))
    add('floor',(0,0,-.1),(10,8,.2),'.65 .67 .68','floor')
    for i, (xyz,size) in enumerate([((0,-4,1.5),(10,.15,3)),((0,4,1.5),(10,.15,3)),((-5,0,1.5),(.15,8,3)),((5,0,1.5),(.15,8,3))]):
        add(f'wall_{i}',xyz,size,'.8 .82 .78','wall')
    # Thin floor details are visual only; physical floor remains level.
    for x in range(-4,5):
        for y in range(-3,4):
            add(f'tile_{x+4}_{y+3}',(x,y,.001),(.92,.92,.002),('.56 .60 .64' if (x+y)%2 else '.72 .73 .69'),'floor',collision=False)
    for i, y in enumerate((-2.7,0,2.7)):
        add(f'gate_panel_{i}',(4.90,y,1.3),(.04,1.6,2.6),'.14 .23 .35','door',textures[i])
        add(f'gate_frame_{i}',(4.85,y,2.65),(.1,1.9,.12),'.15 .15 .15','door')
    for i, x in enumerate((-3,0,3)):
        add(f'poster_south_{i}',(x,-3.90,1.6),(1.8,.05,1.2),'.85 .85 .85','sign',textures[(i+1)%4])
        add(f'window_{i}',(x,3.90,1.7),(1.9,.04,1.4),'.22 .48 .62','window')
        add(f'window_mullion_{i}',(x,3.85,1.7),(.06,.05,1.5),'.1 .12 .14','window')
    add('info_counter',(-3.9,2.5,.5),(1.4,1,1),'.35 .22 .10','furniture',textures[2])
    for i, x in enumerate((-2.8,0,2.8)):
        add(f'bench_{i}',(x,2.6,.42),(1.65,.5,.15),'.20 .22 .28','furniture')
        for j,dx in enumerate((-.65,.65)):
            add(f'bench_leg_{i}_{j}',(x+dx,2.6,.2),(.08,.4,.4),'.12 .12 .12','furniture')
    add('west_sign',(-4.90,0,1.6),(.04,2.2,1.4),'.8 .8 .8','sign',textures[3])
    initial_pose=(-2,-1,.101,0)
    source=None
    if house_path is not None:
        models, objects, source = procthor_structure(Path(house_path), textures)
        initial_pose=(1,6,.101,0)
    world = f'''<?xml version="1.0"?><sdf version="1.9"><world name="{WORLD_NAME}">
      <physics name="physics" type="ignored"><max_step_size>.002</max_step_size><real_time_factor>1</real_time_factor></physics>
      <gravity>0 0 -9.81</gravity><scene><ambient>.55 .55 .55 1</ambient><background>.55 .65 .75 1</background></scene>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
      <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
      <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
      <light name="ceiling" type="directional"><pose>0 0 5 0 0 0</pose><diffuse>.8 .8 .8 1</diffuse>
       <specular>.1 .1 .1 1</specular><direction>.2 .1 -1</direction><cast_shadows>false</cast_shadows></light>
      {''.join(models)}{robot_sdf(initial_pose)}</world></sdf>'''
    sdf = directory/'airport_hall.sdf'; sdf.write_text(world)
    metadata = dict(world=WORLD_NAME, map_id='procthor_scene03000_structure_v1' if source else 'original_airport_hall_v1', coordinate_frame='Gazebo right handed: z up, metres; source positions Unity(x,y,z) mapped to Gazebo(x,z,y); poses are native Gazebo poses, never copied Unity quaternions',
                    source=source, initial_base_pose_xyz_yaw=list(initial_pose),
                    world_sha256=hashlib.sha256(world.encode()).hexdigest(), objects=objects,
                    map_sha256=hashlib.sha256(world.encode()).hexdigest(), world_path=str(sdf),
                    camera=dict(width=WIDTH,height=HEIGHT,K=[[320,0,320],[0,320,240],[0,0,1]],distortion=[0,0,0,0,0]),
                    T_base_camera_optical=T_BASE_CAMERA_OPTICAL,
                    calibration=dict(stereo_baseline_m=.10,rectified=True,
                                     T_left_right=[[1,0,0,.10],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                     T_base_right_camera_optical=[[0,0,1,.20],[-1,0,0,-.10],[0,-1,0,.50],[0,0,0,1]],
                                     synchronization='accept only identical left/right nanosecond sensor timestamps'),
                    sensors=dict(width=WIDTH,height=HEIGHT,hfov_rad=math.pi/2,
                                 K=[[320,0,320],[0,320,240],[0,0,1]], T_base_camera_optical=T_BASE_CAMERA_OPTICAL,
                                 T_base_imu=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                                 wheel_radius_m=WHEEL_RADIUS,wheel_separation_m=WHEEL_SEPARATION),
                    gt_policy='world/model poses only for separate mapping and offline evaluation; never estimator input')
    (directory/'map_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if source:
        (directory/'reference_poses.json').write_text(json.dumps(reference_poses(),indent=2)+'\n')
    return sdf


def _inside(x, y, polygon):
    inside=False
    for (ax,ay),(bx,by) in zip(polygon,polygon[1:]+polygon[:1]):
        if ((ay>y)!=(by>y)) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside=not inside
    return inside


def procthor_structure(house_path, textures):
    """Convert official wall polygons and hole polygons, preserving openings.

    Furniture assets/materials are deliberately not reconstructed. Source room
    polygons and wall/door/window geometry are exact; appearance is original.
    Only axis-aligned source walls are supported, with explicit validation.
    """
    raw=house_path.read_bytes(); house=json.loads(raw)
    models=[]; objects=[]; openings=[]
    def add(name,xyz,size,category,color='.75 .74 .70',texture=None,collision=True):
        if min(size)<=1e-6: return
        models.append(_box(name,xyz,size,color,texture,collision))
        objects.append(dict(name=name,center_m=xyz,size_m=size,category=category,collision=collision))
    for i,r in enumerate(house['rooms']):
        p=[(v['x'],v['z']) for v in r['floorPolygon']]
        xs=sorted(set(x for x,y in p));ys=sorted(set(y for x,y in p))
        for ix,(x0,x1) in enumerate(zip(xs,xs[1:])):
            for iy,(y0,y1) in enumerate(zip(ys,ys[1:])):
                if _inside((x0+x1)/2,(y0+y1)/2,p):
                    add(f'floor_{i}_{ix}_{iy}',((x0+x1)/2,(y0+y1)/2,-.08),(x1-x0,y1-y0,.16),'floor','.60 .64 .66')
                    # Physical floor remains level; repeated decorative tiles.
                    for a in range(math.ceil(x1-x0)):
                        for b in range(math.ceil(y1-y0)):
                            xa=x0+a;ya=y0+b;w=min(.94,x1-xa);h=min(.94,y1-ya)
                            add(f'tile_{i}_{ix}_{iy}_{a}_{b}',(xa+w/2,ya+h/2,.001),(w,h,.002),'floor','.67 .69 .65' if (a+b)%2 else '.49 .53 .55',collision=False)
    bywall={}
    for category in ('doors','windows'):
        for opening in house.get(category,[]):
            for key in ('wall0','wall1'):
                bywall.setdefault(opening[key],[]).append((category[:-1],opening))
    seen=set()
    for wi,w in enumerate(house['walls']):
        if w.get('empty'): continue
        p=w['polygon']; a=p[0]; b=p[1]
        dx=b['x']-a['x'];dy=b['z']-a['z'];length=math.hypot(dx,dy)
        if length<1e-8 or (abs(dx)>1e-6 and abs(dy)>1e-6):
            raise ValueError('Only nondegenerate orthogonal source walls supported')
        key=tuple(sorted(((round(a['x'],6),round(a['z'],6)),(round(b['x'],6),round(b['z'],6)))))
        if key in seen: continue
        seen.add(key);height=max(v['y'] for v in p); holes=[]
        for category,o in bywall.get(w['id'],[]):
            hp=o['holePolygon'];lo=min(q['x'] for q in hp);hi=max(q['x'] for q in hp)
            bottom=min(q['y'] for q in hp);top=max(q['y'] for q in hp)
            if lo < -1e-6 or hi>length+1e-6 or bottom<0 or top>height+1e-6:
                raise ValueError('Source hole outside wall: '+o['id'])
            holes.append((lo,hi,bottom,top))
            openings.append(dict(id=o['id'],category=category,wall_id=w['id'],wall_start=[a['x'],a['z']],wall_end=[b['x'],b['z']],local_bounds=[lo,hi,bottom,top]))
        us=sorted(set([0.,length]+[q for h in holes for q in h[:2]]));zs=sorted(set([0.,height]+[q for h in holes for q in h[2:]]))
        for ui,(u0,u1) in enumerate(zip(us,us[1:])):
            for zi,(z0,z1) in enumerate(zip(zs,zs[1:])):
                u=(u0+u1)/2;z=(z0+z1)/2
                if any(h[0]<u<h[1] and h[2]<z<h[3] for h in holes): continue
                xyz=(a['x']+dx*u/length,a['z']+dy*u/length,z)
                size=(u1-u0,.10,z1-z0) if abs(dx)>1e-6 else (.10,u1-u0,z1-z0)
                # Repeated wall texture is intentionally not a unique pose marker.
                add(f'wall_{wi}_{ui}_{zi}',xyz,size,'wall',texture=textures[wi%4])
    source=dict(path=str(house_path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),dataset='ProcTHOR-10K',scene='scene_03000',official_split='validation',rooms=len(house['rooms']),
                walls_source=len(house['walls']),walls_unique_nonempty=len(seen),openings=openings,
                appearance='original procedural wall textures and floor; source furniture assets omitted; not an AI2-THOR photometric reproduction',
                coordinate_conversion='position(x,y,z)->(x,z,y); do not transform quaternions by simple permutation',
                scope='same known building mapping/query system-development experiment; not held-out SPVLoc network benchmark')
    return models,objects,source


def reference_poses():
    """192 separate mapping viewpoints inside scene03000 open living/kitchen area."""
    return [(x,y,.101,k*math.pi/4) for y in (5.,6.5,8.,9.7)
            for x in (.8,2.,3.5,5.,6.5,7.7) for k in range(8)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--house',type=Path)
    args=p.parse_args(); print(build_world(args.output,args.house))


if __name__ == '__main__': main()
