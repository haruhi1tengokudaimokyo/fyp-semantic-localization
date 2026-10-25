"""READ-ONLY ROS2 bag preflight. Run in system ROS Python; does not run localization or SLAM."""
import sqlite3,json,argparse,hashlib,math
from pathlib import Path
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image as PILImage

def main():
 p=argparse.ArgumentParser();p.add_argument('--bag',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 files=list(Path(a.bag).glob('*.db3'));assert len(files)==1
 db=sqlite3.connect('file:'+str(files[0])+'?mode=ro',uri=True);topics={n:i for i,n in db.execute('SELECT id,name FROM topics')};data={}
 for topic,kind in [('/rgbd/camera_info',CameraInfo),('/rgbd/image',Image)]:
  t,payload=db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1',(topics[topic],)).fetchone();message=deserialize_message(payload,kind)
  if kind==CameraInfo:
   fx=float(message.k[0]);data['camera']={'width':message.width,'height':message.height,'K':list(message.k),'horizontal_fov_deg':float(2*math.atan(message.width/(2*fx))*180/math.pi),'frame_id':message.header.frame_id,'time_ns':t}
  else:
   if message.encoding not in ['rgb8','bgr8']:raise RuntimeError('Unverified encoding '+message.encoding)
   image=np.frombuffer(bytes(message.data),dtype=np.uint8).reshape(message.height,message.step)[:,:message.width*3].reshape(message.height,message.width,3)
   if message.encoding=='bgr8':image=image[:,:,::-1]
   PILImage.fromarray(image).save(out/'first_rgb.png');data['image']={'width':message.width,'height':message.height,'encoding':message.encoding,'frame_id':message.header.frame_id,'time_ns':t}
 for topic in ['/ground_truth','/odom']:
  poses=[]
  for t,payload in db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp',(topics[topic],)):
   m=deserialize_message(payload,Odometry);q=m.pose.pose.orientation;p=m.pose.pose.position;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));poses.append({'time_ns':t,'x':p.x,'y':p.y,'z':p.z,'yaw':yaw,'frame':m.header.frame_id,'child_frame':m.child_frame_id})
  data[topic.strip('/')]=poses
 data['topic_counts']={name:db.execute('SELECT count(*) FROM messages WHERE topic_id=?',(i,)).fetchone()[0] for name,i in topics.items()}
 data['scope']='Existing simulator bag read only; GT and odometry are recorded signals, NOT predictions by this localization system. No temporal localization experiment performed.'
 data['compatibility']='Observed HFOV differs from required 80deg; no naive resize, no independent baseline metric reported.'
 data['source_bag']=a.bag;data['metadata_sha256']=hashlib.sha256((Path(a.bag)/'metadata.yaml').read_bytes()).hexdigest();(out/'preflight.json').write_text(json.dumps(data,indent=2));print(json.dumps({'camera':data['camera'],'counts':data['topic_counts']},indent=2))
if __name__=='__main__':main()
