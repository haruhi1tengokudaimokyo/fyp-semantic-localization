import numpy as np
from evaluation.result_handler import calculate_recalls

def localization_metrics(rows,stage):
    xy=np.array([r[f'{stage}_xy_m'] for r in rows]);yaw=np.array([r[f'{stage}_yaw_deg'] for r in rows])
    return {'n':len(rows),'median_xy_m':float(np.median(xy)),'mean_xy_m':float(xy.mean()),'xy_rmse_m':float(np.sqrt(np.mean(xy**2))),'yaw_mae_deg':float(yaw.mean()),'median_yaw_deg':float(np.median(yaw)),'recall':calculate_recalls(xy,yaw),'severe_error_fraction_xy_gt_3m':float(np.mean(xy>3))}

def classification_metrics(gt,pred,probability):
    gt=np.asarray(gt).ravel();pred=np.asarray(pred).ravel();probability=np.asarray(probability).reshape(-1,4)
    cm=np.zeros((4,4),int);np.add.at(cm,(gt,pred),1)
    rows=[]
    for i,name in enumerate(['wall','window','door','unknown']):
        tp=cm[i,i];support=int(cm[i].sum());count=int(cm[:,i].sum())
        rows.append({'class':name,'gt_count':support,'predicted_count':count,'precision':float(tp/count) if count else None,'recall':float(tp/support) if support else None,'note':'undefined precision/recall represented by null, not fabricated zero'})
    return {'confusion_matrix_gt_rows_pred_columns':cm.tolist(),'classes':rows,'accuracy':float((gt==pred).mean()),'semantic_nll':float(-np.log(np.maximum(probability[np.arange(len(gt)),gt],1e-12)).mean())}

def reliability(raw_scores,success,bins=10):
    """Expose misuse of raw scores as success confidence; NOT a calibration fit."""
    raw_scores=np.asarray(raw_scores);success=np.asarray(success);data=[];ece=0
    for lo in np.arange(bins)/bins:
        pick=(raw_scores>=lo)&(raw_scores<lo+1/bins+(1e-8 if lo>.85 else 0))
        if pick.any():
            mean=float(raw_scores[pick].mean());acc=float(success[pick].mean());ece+=float(pick.mean())*abs(mean-acc);data.append({'lower':float(lo),'n':int(pick.sum()),'mean_raw_score':mean,'actual_joint_success':acc})
    return {'warning':'Counterfactual ECE if heuristic were misused as success probability; no calibrated success probability exists','counterfactual_ece':ece,'bins':data}
