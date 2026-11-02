"""Concrete repairs authorized on 2026-11-01; uses the existing scheduler only."""
def extend_repairs(tasks, repo, base):
    root=repo/'experiment_automation/reproduction'
    def add(tid,paper,python,args,receipt,deps=(),gpu=False,kind='cpu',ram=1000,disk=1,priority=114,equals=None):
        task=dict(task_id=tid,experiment_id='R_'+paper,dataset_id=tid,paper_id=paper,
          reproduction_level='Original Method on Alternative Data' if paper in ['OMCL','SG2Loc'] else 'Hardware Adapted',
          online_input_condition='Declared by each scientific protocol; GT-derived motion explicitly isolated',
          command=[str(base/'envs'/python/'bin/python'),str(root/args[0]),*map(str,args[1:])],
          validation=[dict(path=str(receipt),kind='json',equals=equals or {'status':'accepted'})],
          depends_on=list(deps),inputs=[],formal=False,seed=2089,max_attempts=2,retry_delay_s=120,
          priority=priority,kind=kind,resources=dict(gpu=gpu,cpu_slots=0 if kind=='download' else 1,ram_mib=ram,disk_gib=disk,min_gpu_free_mib=4500),coverage={})
        tasks.append(task);return task
    weights=base/'models/sg2loc_original'
    add('R.SG2Loc.original_checkpoint.acquire','SG2Loc','macvo',['blocked_data.py','sg_checkpoint','pretrained',weights],weights/'checkpoint.json',kind='download',disk=3)
    scweights=base/'models/sg2loc_scannet'
    add('R.SG2Loc.scannet_checkpoint.acquire','SG2Loc','macvo',['blocked_data.py','sg_scannet','epoch31',scweights],scweights/'checkpoint.json',kind='download',disk=1)
    dino=base/'models/torch_hub/checkpoints/dinov2_vitg14_pretrain.pth'
    add('R.SG2Loc.dino_weights.acquire','SG2Loc','macvo',['acquire_public.py','https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',dino],dino.with_suffix('.pth.json'),kind='download',disk=5,equals={'zip_crc_verified':False})
    add('R.SG2Loc.dino_weights.resume','SG2Loc','macvo',['resume_publisher_download.py','https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth',dino],dino.with_suffix('.pth.json'),kind='download',disk=3,priority=130,equals={'zip_crc_verified':False})
    roma=base/'models/torch_hub/checkpoints/romav2.pt'
    add('R.SG2Loc.roma_weights.acquire','SG2Loc','macvo',['resume_publisher_download.py','https://github.com/Parskatt/RoMaV2/releases/download/weights/romav2.pt',roma],roma.with_suffix('.pt.json'),kind='download',disk=2,priority=130,equals={'zip_crc_verified':False})
    core_metrics={'SG2Loc':[],'OMCL':[]};core_deps={'SG2Loc':[],'OMCL':[]}
    for scene in ['v3_sc0_staging_00','v3_sc1_staging_00']:
        data=base/'datasets/replicacad_localization'/scene
        old_render=add('R.data.replicacad.'+scene+'.render','OMCL','habitat',['replicacad_sequence.py',scene,data],data/'acceptance.json',gpu=True,ram=1500,disk=2,priority=125)
        old_render['blocked_reason']='Two failed attempts preserved: missing import then PBR constant material conversion; all 600+128 observations already rendered. Reuse them in mesh_finish CPU repair, no repeated rendering.'
        ready=add('R.data.replicacad.'+scene+'.mesh_finish','OMCL','habitat',['replicacad_sequence.py',scene,data,'--mesh-only'],data/'acceptance.json',ram=2000,disk=1,priority=128)
        out=base/'runs/reproduction/OMCL/ReplicaCAD'/scene
        dep=ready['task_id']
        f=add('R.OMCL.'+scene+'.features','OMCL','omcl',['omcl_replicacad.py','features',data,out],out/'features.json',[dep],gpu=True,ram=5000,disk=2,priority=124)
        mp=add('R.OMCL.'+scene+'.map','OMCL','omcl',['omcl_replicacad.py','map',data,out],out/'map.json',[f['task_id']],gpu=True,ram=2500,disk=1,priority=124)
        for mode in ['tracking','global','zero_motion_noise']:
            t=add('R.OMCL.'+scene+'.'+mode,'OMCL','omcl',['omcl_replicacad.py',mode,data,out],out/mode/'execution.json',[mp['task_id']],gpu=True,ram=2300,disk=1,priority=123,equals={'status':'executed','localization_executed':True,'frames':600})
            t['online_input_condition']='Noisy GT-derived relative motion; tracking initialized near GT, global initialized using map-only references. Not GT-free.'
            metric=add(t['task_id']+'.metrics','OMCL','macvo',['localization_metrics.py',data,out/mode],out/mode/'metrics.json',[t['task_id']],equals={'status':'accepted','localization_executed':True,'independently_recomputed':True})
            if mode=='global':core_metrics['OMCL'].append(out/mode/'metrics.json');core_deps['OMCL'].append(metric['task_id'])
        sg=base/'runs/reproduction/SG2Loc/ReplicaCAD'/scene
        motion=add('R.SG2Loc.'+scene+'.measured_motion','SG2Loc','macvo',['replicacad_rgbd_motion.py',data,sg],sg/'motion.json',[dep],ram=900,priority=126,equals={'status':'accepted','GT_online':False,'frames':600})
        geom=add('R.SG2Loc.'+scene+'.geometry','SG2Loc','sg2loc',['sg2loc_replicacad.py','geometry',data,sg],sg/'geometry.json',[dep],ram=2200,disk=1,priority=126)
        ft=add('R.SG2Loc.'+scene+'.features','SG2Loc','sg2loc',['sg2loc_replicacad.py','features',data,sg],sg/'features.json',[geom['task_id'],'R.SG2Loc.dino_weights.resume','R.SG2Loc.scannet_checkpoint.acquire'],gpu=True,ram=7000,disk=2,priority=126)
        for length in [5,10,25]:
            pf=add('R.SG2Loc.'+scene+'.length'+str(length),'SG2Loc','sg2loc',['sg2loc_replicacad.py','localize',data,sg,'--length',length],sg/f'length{length}/execution.json',[ft['task_id'],motion['task_id']],gpu=True,ram=2700,disk=1,priority=126,equals={'status':'executed','localization_executed':True,'frames':600,'GT_online':False})
            rf=add(pf['task_id']+'.refinement','SG2Loc','sg2loc',['sg2loc_replicacad.py','refine',data,sg,'--length',length],sg/f'length{length}/refinement/execution.json',[pf['task_id'],'R.SG2Loc.roma_weights.acquire'],gpu=True,ram=4000,disk=2,priority=126,equals={'status':'executed','localization_executed':True,'refinement_executed':True,'frames':600})
            mt=add(rf['task_id']+'.metrics','SG2Loc','macvo',['localization_metrics.py',data,sg/f'length{length}/refinement'],sg/f'length{length}/refinement/metrics.json',[rf['task_id']],equals={'status':'accepted','independently_recomputed':True,'localization_executed':True})
            if length==5:core_metrics['SG2Loc'].append(sg/f'length{length}/refinement/metrics.json');core_deps['SG2Loc'].append(mt['task_id'])
    # Replace legacy empty-command IDs with actual acceptance commands. These
    # remain dependency-blocked until real localization/refinement has finished.
    for paper in ['SG2Loc','OMCL']:
        for suffix in ['official_benchmark','replica_core']:
            tid='R.'+paper+'.'+suffix
            old=next(t for t in tasks if t['task_id']==tid)
            result=base/'runs/reproduction'/paper/('replicacad_'+suffix+'_acceptance.json')
            old['command']=[str(base/'envs/macvo/bin/python'),str(root/'core_result_acceptance.py'),str(result),*map(str,core_metrics[paper])]
            old['depends_on']=core_deps[paper];old.pop('blocked_reason',None)
            old['inputs']=[];old['dataset_id']='ReplicaCAD_shared_FRL_layouts'
            old['reproduction_level']='Original Method on Alternative Data'
            old['online_input_condition']='Measured RGB-D motion, level camera mounting; query GT isolated' if paper=='SG2Loc' else 'Explicit original-paper GT-derived relative-motion condition; not GT-free online'
            old['validation']=[dict(path=str(result),kind='json',equals={'status':'accepted','localization_executed':True,'independent_metrics':True,'not_original_paper_dataset':True})]
            old['config']={'legacy_task_name':True,'actual_scope':'ReplicaCAD layouts, not original ScanNet/3RScan/Replica paper benchmark'}
    for environment in ['abandonedfactory','hospital']:
        ds=base/'datasets/tartanair_full'/environment/'P000'
        download=add('R.MACVO.'+environment+'.P000.acquire','MACVO','macvo',['acquire_tartan_full.py',environment,'P000',ds],ds/'acceptance.json',kind='download',disk=5,priority=118)
        output=base/'runs/reproduction/MACVO'/environment/'P000/paper'
        run=add('R.MACVO.'+environment+'.P000.paper','MACVO','macvo',['macvo_full_sequence.py',ds/'sequence',output,'--source',base/'sources/MAC-VO','--weights',base/'models/MACVO_FrontendCov.pth','--config','paper','--motion-weights',base/'models/MACVO_posenet.pkl','--decoder','fp32','--encoder','fp32'],output/'summary.json',[download['task_id']],gpu=True,ram=3000,disk=2,priority=120,equals={'initialization_included':True})
        run['formal']=True;run['online_input_condition']='Real publisher full stereo sequence only; GT loaded after online run, never as motion input'
        add(run['task_id']+'.official_metrics','MACVO','macvo',['macvo_new_metrics.py',ds,output],output/'official_metrics.json',[run['task_id']],equals={'status':'accepted','full_source_sequence':True,'GT_online':False,'original_metric_functions':True})
        ab=output.parent/'weight_ablation'
        add(run['task_id']+'.weight_ablation','MACVO','macvo',[str(repo/'research3d/scripts/replay_macvo_weights.py'),output,ab,'--source',base/'sources/MAC-VO'],ab/'summary.json',[run['task_id']],ram=2500,priority=120,equals={'audit.GT_used_during_solve':False,'audit.identity_applied_inside_actual_solver':True})
        icp=output.parent/'original_icp_weight_ablation'
        add(run['task_id']+'.icp_weight_ablation','MACVO','macvo',['macvo_icp_ablation.py',output,icp,'--source',base/'sources/MAC-VO'],icp/'summary.json',[run['task_id']],ram=2500,priority=120,equals={'audit.GT_used_during_solve':False,'audit.identity_applied_inside_actual_solver':True})
    for name in ['MH_01_easy','MH_02_easy']:
        ds=base/'datasets/euroc_full'/name
        dl=add('R.MACVO.EuRoC.'+name+'.acquire','MACVO','macvo',['acquire_euroc_full.py',name,ds],ds/'acceptance.json',kind='download',disk=5,priority=117)
        out=base/'runs/reproduction/MACVO/EuRoC'/name/'paper'
        run=add('R.MACVO.EuRoC.'+name+'.paper','MACVO','macvo',['macvo_full_sequence.py',ds/'mav0',out,'--dataset-kind','euroc','--source',base/'sources/MAC-VO','--weights',base/'models/MACVO_FrontendCov.pth','--config','paper','--motion-weights',base/'models/MACVO_posenet.pkl','--decoder','fp32','--encoder','fp32'],out/'summary.json',[dl['task_id']],gpu=True,ram=3200,disk=2,priority=120,equals={'initialization_included':True})
        run['formal']=True;run['online_input_condition']='Official EuRoC rectified stereo only, no online IMU/GT. All synchronized RGB frames; GT overlap scored offline.'
        add(run['task_id']+'.official_metrics','MACVO','macvo',['macvo_new_metrics.py',ds,out],out/'official_metrics.json',[run['task_id']],equals={'status':'accepted','full_source_sequence':True,'GT_online':False,'original_metric_functions':True})
    for name in ['cafe1-1','cafe1-2']:
        ds=base/'datasets/openloris_full'/name;out=base/'runs/reproduction/MINS'/name
        dl=add('C.full.'+name+'.acquire','MINS','sensors',['openloris_full.py','acquire',name,ds],ds/'acceptance.json',kind='download',ram=400,disk=5,priority=116)
        dl=add('C.full.'+name+'.acquire.v2','MINS','sensors',['openloris_full.py','acquire',name,ds],ds/'acceptance.json',kind='download',ram=400,disk=5,priority=116)
        dl['config']={'repair':'Actual requests import failure fixed in isolated sensors environment; failed attempts retained, same full source bag'}
        prep=add('C.full.'+name+'.prepare','MINS','sensors',[str(repo/'research3d/scripts/prepare_openloris_mins.py'),ds/(name+'.bag'),out,'--source',base/'sources/MINS'],out/'audit.json',[dl['task_id']],ram=3500,disk=3,equals={'GT_online':False})
        audit=add('C.full.'+name+'.coverage','MINS','sensors',['openloris_full.py','coverage',ds/(name+'.bag'),out,out/'full_coverage.json'],out/'full_coverage.json',[prep['task_id']],ram=500,equals={'status':'accepted','full_source_bag':True,'all_camera_messages_preserved':True,'GT_online':False})
        run=add('C.full.'+name+'.stereo_imu','MINS','sensors',[str(repo/'research3d/scripts/run_mins_suite.py'),out,'--binary',base/'build/mins/mins_bag','--modes','stereo_imu'],out/'execution.json',[audit['task_id']],ram=2500,equals={'stereo_imu.returncode':0,'stereo_imu.use_gt':False})
        add(run['task_id']+'.metrics','MINS','sensors',[str(repo/'research3d/scripts/evaluate_mins.py'),out,'--modes','stereo_imu'],out/'comparison.json',[run['task_id']],ram=600,equals={'stereo_imu.protocol.GT_online':False})
    # Apply only to new, not-yet-run GPU tasks; completed geometry is immutable.
    kernel=next(t for t in tasks if t['task_id']=='R.SG2Loc.kernel_probe')
    for t in tasks:
        if t['task_id'].startswith('R.SG2Loc.v3_') and t['resources']['gpu']:
            t['env']=dict(kernel['env'])
    # The originally empty core IDs now execute the actual upstream algorithm,
    # with independent trajectory metrics required before COMPLETED. Preserve
    # their already-registered unexecuted aliases as held, avoiding duplicate GPU work.
    import copy
    for paper,alias,phase in [
        ('OMCL','R.OMCL.v3_sc0_staging_00.global','global_accept'),
        ('SG2Loc','R.SG2Loc.v3_sc0_staging_00.length5.refinement','refine_accept')]:
        source=next(t for t in tasks if t['task_id']==alias)
        core=next(t for t in tasks if t['task_id']==f'R.{paper}.replica_core')
        updated=copy.deepcopy(source);updated['task_id']=core['task_id'];updated['command'][2]=phase
        for check in updated['validation']:
            check['path']=check['path'].replace('execution.json','metrics.json')
            check['equals']={'status':'accepted','localization_executed':True,'independently_recomputed':True,'frames':600}
        updated['dataset_id']='ReplicaCAD_v3_sc0_staging_00'
        core.clear();core.update(updated)
        for task in tasks:
            if task is source:continue
            task['depends_on']=[core['task_id'] if dep==alias else dep for dep in task['depends_on']]
        source['blocked_reason']='Unexecuted alias promoted to '+core['task_id']+'; do not duplicate GPU inference'
    # Public scene acquisition must not depend on SG2Loc success: OMCL expansion
    # remains runnable if the independent SG2Loc branch fails.
    for scene in ['office_0','room_0']:
        assets=base/'datasets/replica_independent'/scene
        dl=add('R.data.Replica.'+scene+'.acquire','SG2Loc','macvo',['acquire_replica_scene.py',scene,assets],assets/'acquisition.json',['R.OMCL.replica_core'],kind='download',ram=700,disk=3,priority=127)
        data=base/'datasets/replica_independent_sequences'/scene
        render=add('R.data.Replica.'+scene+'.render','SG2Loc','habitat',['replica_sequence.py',scene,data],data/'acceptance.json',[dl['task_id']],gpu=True,ram=2500,disk=3,priority=127)
        sg=base/'runs/reproduction/SG2Loc/Replica'/scene
        geometry=add('R.SG2Loc.Replica.'+scene+'.geometry','SG2Loc','sg2loc',['sg2loc_replicacad.py','geometry',data,sg],sg/'geometry.json',[render['task_id']],ram=3500,disk=2,priority=127)
        motion=add('R.SG2Loc.Replica.'+scene+'.motion','SG2Loc','macvo',['replicacad_rgbd_motion.py',data,sg],sg/'motion.json',[render['task_id']],ram=900,priority=127,equals={'status':'accepted','GT_online':False,'frames':600})
        ft=add('R.SG2Loc.Replica.'+scene+'.features','SG2Loc','sg2loc',['sg2loc_replicacad.py','features',data,sg],sg/'features.json',[geometry['task_id'],'R.SG2Loc.dino_weights.resume','R.SG2Loc.scannet_checkpoint.acquire'],gpu=True,ram=7000,disk=3,priority=126)
        for length in [5,10,25]:
            pf=add('R.SG2Loc.Replica.'+scene+f'.length{length}','SG2Loc','sg2loc',['sg2loc_replicacad.py','localize',data,sg,'--length',length],sg/f'length{length}/execution.json',[ft['task_id'],motion['task_id']],gpu=True,ram=2800,disk=1,priority=126,equals={'status':'executed','localization_executed':True,'GT_online':False,'frames':600})
            rf=add(pf['task_id']+'.refinement','SG2Loc','sg2loc',['sg2loc_replicacad.py','refine_accept',data,sg,'--length',length],sg/f'length{length}/refinement/metrics.json',[pf['task_id'],'R.SG2Loc.roma_weights.acquire'],gpu=True,ram=4200,disk=2,priority=126,equals={'status':'accepted','localization_executed':True,'independently_recomputed':True,'frames':600})
            pf['env']=dict(kernel['env']);rf['env']=dict(kernel['env'])
        om=base/'runs/reproduction/OMCL/Replica'/scene
        feat=add('R.OMCL.Replica.'+scene+'.features','OMCL','omcl',['omcl_replicacad.py','features',data,om],om/'features.json',[render['task_id']],gpu=True,ram=5000,disk=2,priority=124)
        mp=add('R.OMCL.Replica.'+scene+'.map','OMCL','omcl',['omcl_replicacad.py','map',data,om],om/'map.json',[feat['task_id']],gpu=True,ram=3000,disk=1,priority=124)
        for mode in ['tracking','global','zero_motion_noise']:
            run=add('R.OMCL.Replica.'+scene+'.'+mode,'OMCL','omcl',['omcl_replicacad.py',mode,data,om],om/mode/'execution.json',[mp['task_id']],gpu=True,ram=2500,disk=1,priority=123,equals={'status':'executed','localization_executed':True,'frames':600})
            add(run['task_id']+'.metrics','OMCL','macvo',['localization_metrics.py',data,om/mode],om/mode/'metrics.json',[run['task_id']],equals={'status':'accepted','independently_recomputed':True,'frames':600})
    # Finish one SG2Loc core trajectory, then the prepared other methods before
    # further SG2Loc ablations. These scheduling priorities never preempt a worker
    # or alter scientific configurations of accepted outputs.
    for task in tasks:
        tid=task['task_id']
        if tid=='R.SG2Loc.v3_sc0_staging_00.length5':task['priority']=130
        elif tid=='R.SG2Loc.replica_core':task['priority']=131
        elif tid.startswith('R.MACVO.') and 'P000.paper' in tid:task['priority']=129
        elif tid.startswith('R.OMCL.') and task.get('resources',{}).get('gpu'):task['priority']=128
        elif tid.startswith('R.ACE.7scenes.') and task.get('resources',{}).get('gpu'):task['priority']=127
        elif tid.startswith('R.SG2Loc.') and task.get('resources',{}).get('gpu'):task['priority']=116
    from macvo_followup_manifest import extend
    return extend(tasks,repo,base)
