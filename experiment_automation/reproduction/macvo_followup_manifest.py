"""2026-11-02 authorized MAC-VO ablation/runtime investigation; old results intact."""
def extend(tasks,repo,base):
    scripts=repo/'experiment_automation/reproduction';python=str(base/'envs/macvo/bin/python')
    def add(tid,args,checks,deps=(),gpu=False,priority=150,disk=1):
        spec=dict(task_id=tid,experiment_id='R_MACVO',paper_id='MACVO',dataset_id=tid,
            reproduction_level='Hardware Adapted',online_input_condition='Stereo RGB + calibration only; actual original TartanMotionNet, GT read only after trajectory completes',
            command=[python,str(scripts/args[0]),*map(str,args[1:])],depends_on=list(deps),inputs=[],seed=1305,
            formal=False,max_attempts=1,retry_delay_s=120,priority=priority,kind='cpu',
            resources=dict(gpu=gpu,cpu_slots=4 if gpu else 1,ram_mib=3000 if gpu else 1600,disk_gib=disk,min_gpu_free_mib=5200),
            validation=[dict(path=str(path),kind='json',equals=equals) for path,equals in checks],coverage={})
        tasks.append(spec);return tid
    proof=base/'runs/reproduction/MACVO/diagnostics_20261102/solver_contract'
    contract=add('R.MACVO.solver_spawn_contract.v1',['macvo_solver_contract.py',base/'sources/MAC-VO',base/'runs/reproduction/MACVO/hospital/P000/paper',proof],[(proof/'acceptance.json',dict(status='accepted',spawn_verified=True,actual_original_ICP=True,GT_read=False))])
    for environment,first_priority in [('hospital',151),('abandonedfactory',147)]:
        dataset=base/'datasets/tartanair_full'/environment/'P000'
        output=base/'runs/reproduction/MACVO/diagnostics_20261102'/environment
        metrics=[];ablation_metrics=[];precision_metrics=[]
        for index,(mode,covariance,precision) in enumerate([('original_fp32','original','fp32'),('original_fp16','original','fp16'),('identity_fp32','identity','fp32')]):
            out=output/mode;tid=f'R.MACVO.{environment}.P000.audit.{mode}.v1'
            run=add(tid,['macvo_instrumented_sequence.py',dataset/'sequence',out,'--source',base/'sources/MAC-VO','--weights',base/'models/MACVO_FrontendCov.pth','--config','paper','--motion-weights',base/'models/MACVO_posenet.pkl','--encoder',precision,'--decoder',precision,'--backend-covariance',covariance,'--capture-graphs'],[(out/'summary.json',{'initialization_included':True}), (out/'stage_profile.json',{'initialization_and_final_optimizer_drain_included':True,'per_frame_global_sync':False})],[contract,f'R.MACVO.{environment}.P000.acquire'],gpu=True,priority=first_priority-index)
            if mode=='original_fp16' and environment=='abandonedfactory':
                tasks[-1]['blocked_reason']='Hospital FP16 trial produced zero observations for 25 consecutive frames and was slower; compatibility not passed. Preserve command, do not propagate invalid precision to another scene.'
            if mode=='original_fp32':
                add(tid+'.exact_replay',['macvo_exact_solver_replay.py',out,base/'sources/MAC-VO'],[(out/'exact_solver_replay.json',dict(status='accepted',full_saved_solver_job_coverage=True,motion_initialization_preserved=True,GT_read=False))],[run],priority=160)
            metrics.append(add(tid+'.metrics',['macvo_new_metrics.py',dataset,out],[(out/'official_metrics.json',dict(status='accepted',full_source_sequence=True,GT_online=False,original_metric_functions=True))],[run],priority=160))
            if mode in ['original_fp32','identity_fp32']:ablation_metrics.append(metrics[-1])
            if mode in ['original_fp32','original_fp16']:precision_metrics.append(metrics[-1])
        add(f'R.MACVO.{environment}.P000.audit.comparison.v1',['macvo_followup_acceptance.py',output,'--section','ablation'],[(output/'comparison.json',dict(status='accepted',full_pipeline_ablation_executed=True,all_metrics_independent=True))],ablation_metrics,priority=160)
        add(f'R.MACVO.{environment}.P000.audit.precision_comparison.v1',['macvo_followup_acceptance.py',output,'--section','precision'],[(output/'precision_comparison.json',dict(status='accepted',all_metrics_independent=True))],precision_metrics,priority=160)
    return tasks
