"""Synthetic contract tests; these are not ACE or localization benchmark results."""
import json
import math
from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from research3d.robot_sim.localization import (GlobalLocalizer,LocalizationConfig,
    OdometryHistory,differential_wheel_velocity)
from research3d.robot_sim.scheduler import LatestOnlyWorker,QueryTrigger,AceRpcProcess,ExecutionAlternator


EXTRINSIC=np.array([[0,0,1,.2],[-1,0,0,0],[0,-1,0,.5],[0,0,0,1]],dtype=float)


def se2(x=0,y=0,yaw=0):
    t=np.eye(4)
    t[:3,:3]=Rotation.from_euler("z",yaw).as_matrix()
    t[:3,3]=[x,y,0]
    return t


class LocalizationTests(unittest.TestCase):
    def core(self,**kwargs):
        defaults=dict(map_id="buildingA",confirmations=2,confirmation_span_s=.1)
        defaults.update(kwargs)
        return GlobalLocalizer(LocalizationConfig(**defaults),EXTRINSIC)

    def history_until(self,core,now):
        last=core.history.samples[-1][0] if core.history.samples else -.05
        for t in np.arange(last+.05,now+.001,.05):
            core.add_odometry(round(float(t),6),se2(x=t))

    def observation(self,t,offset=10.,qid=None,**kwargs):
        row=dict(status="ok",query_id=qid or str(t),stamp_sec=t,map_id="buildingA",
                 source="official_ace_dsacstar",inliers=150,reprojection_error_px=None,
                 T_map_camera_optical=(se2(offset)@se2(t)@EXTRINSIC).tolist())
        row.update(kwargs)
        return row

    def initialize(self,core):
        for stamp in [.1,.3]:
            self.history_until(core,stamp+.1)
            core.observe(self.observation(stamp),stamp+.1)

    def test_delayed_camera_extrinsic_and_current_propagation(self):
        core=self.core()
        self.initialize(core)
        self.history_until(core,.7)
        state=core.snapshot(.7)
        self.assertTrue(state["global_valid"])
        np.testing.assert_allclose(state["T_map_odom"],se2(10),atol=1e-10)
        np.testing.assert_allclose(state["T_map_base"],se2(10.7),atol=1e-10)
        np.testing.assert_allclose(state["T_odom_base"],se2(.7),atol=1e-10)

    def test_local_never_depends_on_absolute_success(self):
        core=self.core()
        self.history_until(core,.5)
        result=core.observe(self.observation(.4,status="failed"),.5)
        self.assertEqual(result["reason"],"localizer_failed")
        self.assertTrue(core.snapshot(.5)["local_valid"])
        self.assertFalse(core.snapshot(.5)["global_valid"])

    def test_map_mismatch_no_mutation(self):
        core=self.core()
        self.history_until(core,.5)
        result=core.observe(self.observation(.4,map_id="wrong-building"),.5)
        self.assertEqual(result["reason"],"map_mismatch")
        self.assertEqual(core.hypotheses,[])

    def test_truth_fields_rejected(self):
        core=self.core()
        self.history_until(core,.5)
        self.assertIn("invalid_contract",core.observe(self.observation(.4,gt_pose=[0,0,0]),.5)["reason"])

    def test_same_map_name_wrong_map_revision_rejected(self):
        core=self.core(map_sha256="a"*64)
        self.history_until(core,.5)
        self.assertEqual(core.observe(self.observation(.4,map_sha256="b"*64),.5)["reason"],"map_fingerprint_mismatch")

    def test_stale_observation_cannot_correct(self):
        core=self.core(max_observation_age_s=.3)
        self.history_until(core,.8)
        self.assertEqual(core.observe(self.observation(.1),.8)["reason"],"future_or_stale_observation")
        self.assertIsNone(core.T_map_odom)

    def test_future_timestamp_does_not_poison_valid_queue(self):
        core=self.core()
        self.history_until(core,.5)
        core.observe(self.observation(100.),.5)
        self.assertEqual(core.observe(self.observation(.4),.5)["reason"],"awaiting_temporal_confirmation")

    def test_duplicate_not_independent_confirmation(self):
        core=self.core()
        self.history_until(core,.5)
        core.observe(self.observation(.2),.5)
        result=core.observe(self.observation(.2),.5)
        self.assertEqual(result["reason"],"duplicate_or_out_of_order")
        self.assertIsNone(core.T_map_odom)

    def test_kidnap_recovery_can_leave_old_global_pose(self):
        core=self.core()
        self.initialize(core)
        for stamp in [.6,.8,1.0]:
            self.history_until(core,stamp+.1)
            core.observe(self.observation(stamp,offset=30),stamp+.1)
        self.assertEqual(core.last_decision["reason"],"confirmed_global_recovery")
        np.testing.assert_allclose(core.T_map_odom,se2(30),atol=1e-10)
        np.testing.assert_allclose(core.history.samples[-1][1],se2(1.1),atol=1e-10)

    def test_distant_modes_not_averaged(self):
        core=self.core(confirmations=3)
        for stamp,offset in [(.1,10),(.3,30),(.5,10),(.7,30)]:
            self.history_until(core,stamp+.1)
            core.observe(self.observation(stamp,offset),stamp+.1)
        self.assertIsNone(core.T_map_odom)
        self.assertEqual(len(core.hypotheses),2)
        self.assertEqual(sorted(round(h["transform"][0,3]) for h in core.hypotheses),[10,30])

    def test_low_inlier_does_not_bootstrap(self):
        core=self.core()
        self.history_until(core,.5)
        self.assertEqual(core.observe(self.observation(.4,inliers=99),.5)["reason"],"insufficient_inliers")

    def test_planar_height_rejects_camera_frame_error(self):
        core=self.core()
        self.history_until(core,.5)
        wrong=se2(10.4)
        wrong[2,3]=2.
        result=core.observe(self.observation(.4,T_map_camera_optical=wrong.tolist()),.5)
        self.assertEqual(result["reason"],"implausible_planar_robot_pose")

    def test_dead_reckoning_expiry_preserves_local(self):
        core=self.core(max_dead_reckoning_s=.5)
        self.initialize(core)
        self.history_until(core,1.)
        state=core.snapshot(1.)
        self.assertTrue(state["local_valid"])
        self.assertFalse(state["global_valid"])
        self.assertEqual(state["global_state"],"RECOVERING")

    def test_missing_local_stream_not_valid_global(self):
        core=self.core()
        self.initialize(core)
        self.assertEqual(core.snapshot(1.)["local_state"],"STALE")
        self.assertFalse(core.snapshot(1.)["global_valid"])

    def test_degraded_absolute_covariance_does_not_deadlock_recovery(self):
        core=self.core(max_local_covariance_trace=.01)
        for stamp in [.1,.3]:
            self.history_until(core,stamp+.1)
            core.history.samples[-1]=(core.history.samples[-1][0],core.history.samples[-1][1],1000.)
            self.assertEqual(core.snapshot(stamp+.1)["local_state"],"DEGRADED")
            self.assertTrue(core.snapshot(stamp+.1)["local_valid"])
            core.observe(self.observation(stamp),stamp+.1)
        self.assertTrue(core.snapshot(.4)["global_valid"])
        trigger=QueryTrigger()
        rgb=np.arange(48,dtype=np.uint8).reshape(4,4,3)*5
        self.assertTrue(trigger.decide(rgb,.4,core.snapshot(.4))[0])

    def spv(self,stamp,offsets):
        return dict(query_id=str(stamp),stamp_sec=stamp,map_id="buildingA",source="official_spvloc",status="ok",
            candidates=[dict(rank=i+1,viewport_score=.7-i*.1,
                T_map_camera_optical=(se2(offset)@se2(stamp)@EXTRINSIC).tolist()) for i,offset in enumerate(offsets)])

    def test_spvloc_unique_temporal_mode_can_be_rank_two(self):
        core=self.core(allow_spvloc_corrections=True)
        for stamp,offsets in [(.1,[20,10,30]),(.3,[22,10,32])]:
            self.history_until(core,stamp+.1)
            decision=core.observe_candidates(self.spv(stamp,offsets),stamp+.1)
        self.assertEqual(decision["reason"],"confirmed_spvloc_recovery")
        np.testing.assert_allclose(core.T_map_odom,se2(10),atol=1e-10)

    def test_spvloc_multiple_coherent_modes_abstain(self):
        core=self.core(allow_spvloc_corrections=True)
        for stamp in [.1,.3]:
            self.history_until(core,stamp+.1)
            decision=core.observe_candidates(self.spv(stamp,[10,30]),stamp+.1)
        self.assertEqual(decision["reason"],"ambiguous_spvloc_recovery")
        self.assertIsNone(core.T_map_odom)

    def test_spvloc_near_duplicate_top_three_are_one_observation(self):
        core=self.core(allow_spvloc_corrections=True)
        self.history_until(core,.2)
        core.observe_candidates(self.spv(.1,[10,10.1,10.2]),.2)
        self.assertIsNone(core.T_map_odom)
        self.assertEqual(len(core.hypotheses),1)
        self.assertEqual(len(core.hypotheses[0]["observations"]),1)

    def test_unqualified_spvloc_is_audited_without_disrupting_ace_confirmation(self):
        core=self.core(confirmations=3,confirmation_span_s=1.)
        for stamp in [.1,.6,1.1,1.6,2.1]:
            self.history_until(core,stamp+.1)
            if stamp in [.6,1.6]:
                observations=[list(h["observations"]) for h in core.hypotheses]
                transforms=[h["transform"].copy() for h in core.hypotheses]
                decision=core.observe_candidates(self.spv(stamp,[100,200,300]),stamp+.1)
                self.assertEqual(decision["reason"],"spvloc_shadow_unqualified_model")
                self.assertIsNone(core.T_map_odom)
                self.assertEqual([h["observations"] for h in core.hypotheses],observations)
                for h,old in zip(core.hypotheses,transforms):np.testing.assert_array_equal(h["transform"],old)
            else:decision=core.observe(self.observation(stamp),stamp+.1)
        self.assertTrue(decision["accepted"])
        np.testing.assert_allclose(core.T_map_odom,se2(10),atol=1e-10)
        # Even a later persistent alternative may not change an established map.
        self.history_until(core,2.7)
        decision=core.observe_candidates(self.spv(2.6,[100,200,300]),2.7)
        self.assertEqual(decision["reason"],"spvloc_shadow_unqualified_model")
        np.testing.assert_allclose(core.T_map_odom,se2(10),atol=1e-10)
        self.assertFalse(core.snapshot(2.7)["spvloc_corrections_allowed"])

    def test_tracking_gate_is_invariant_to_arbitrary_odom_origin(self):
        for origin in [0.,10.,1000.]:
            core=self.core()
            for stamp in [0.,.1,.2,.3]:core.add_odometry(stamp,se2(origin+10))
            core.T_map_odom=se2(-origin)
            core.last_accepted_stamp=0.
            core.global_state="TRACKING"
            # Same map position, legitimate ten-degree rotation correction.
            candidate=se2(10,yaw=math.radians(10))@EXTRINSIC
            row=self.observation(.2,T_map_camera_optical=candidate.tolist())
            result=core.observe(row,.3)
            self.assertTrue(result["accepted"],(origin,result))
            np.testing.assert_allclose(core.snapshot(.3)["T_map_base"],se2(10,yaw=math.radians(10)),atol=1e-10)

    def test_recovery_gate_is_invariant_to_odom_origin(self):
        for origin in [0.,1000.]:
            core=self.core()
            for stamp in [0.,.1,.2,.3]:core.add_odometry(stamp,se2(origin+10))
            first=self.observation(.1,T_map_camera_optical=(se2(10)@EXTRINSIC).tolist())
            second=self.observation(.3,T_map_camera_optical=(se2(10,yaw=math.radians(10))@EXTRINSIC).tolist())
            core.observe(first,.3)
            result=core.observe(second,.3)
            self.assertTrue(result["accepted"],(origin,result))

    def test_interpolation_slerp_and_no_extrapolation(self):
        history=OdometryHistory(max_gap_s=.2)
        history.append(1.,se2(0,yaw=0))
        history.append(1.1,se2(1,yaw=math.pi/2))
        np.testing.assert_allclose(history.at(1.05),se2(.5,yaw=math.pi/4),atol=1e-12)
        with self.assertRaises(ValueError):history.at(1.11)
        with self.assertRaises(ValueError):history.append(1.1,se2())

    def test_missing_local_interval_not_interpolated(self):
        history=OdometryHistory(max_gap_s=.15)
        history.append(1.,se2())
        history.append(2.,se2(1))
        with self.assertRaisesRegex(ValueError,"missing local odometry"):history.at(1.5)

    def test_wheel_kinematics_sign(self):
        self.assertEqual(differential_wheel_velocity(2.,2.,.1,.42),(.2,0.))
        forward,yaw=differential_wheel_velocity(-1.,1.,.1,.42)
        self.assertAlmostEqual(forward,0.)
        self.assertGreater(yaw,0)


class SchedulingTests(unittest.TestCase):
    def test_actual_execution_alternation_survives_latest_pending_replacements(self):
        selector=ExecutionAlternator(True,True)
        entered=[threading.Event() for _ in range(3)]
        release=[threading.Event() for _ in range(3)]
        done=threading.Event()
        branches=[]
        outputs=[]
        def work(item):
            i=len(branches)
            branches.append(selector.choose(recovering=True))
            entered[i].set()
            self.assertTrue(release[i].wait(3))
            return item
        def result(item):
            outputs.append(item)
            if len(outputs)==3:done.set()
        worker=LatestOnlyWorker(work,result)
        try:
            worker.submit({"request_id":"initial"})
            self.assertTrue(entered[0].wait(3))
            for i in range(12):worker.submit({"request_id":f"replaced_first_{i}"})
            release[0].set();self.assertTrue(entered[1].wait(3))
            for i in range(13):worker.submit({"request_id":f"replaced_second_{i}"})
            release[1].set();self.assertTrue(entered[2].wait(3))
            release[2].set();self.assertTrue(done.wait(3))
            self.assertEqual(branches,["ace","spvloc","ace"])
            self.assertEqual([x["request_id"] for x in outputs],["initial","replaced_first_11","replaced_second_12"])
            self.assertEqual(selector.choose(recovering=False),"ace")
        finally:
            for event in release:event.set()
            worker.close()

    def test_single_worker_replaces_pending_without_parallel_calls(self):
        started=threading.Event()
        release=threading.Event()
        done=threading.Event()
        calls=[]
        results=[]
        def work(item):
            calls.append(item["request_id"])
            if len(calls)==1:
                started.set()
                self.assertTrue(release.wait(2))
            return item
        def result(item):
            results.append(item)
            if len(results)==2:done.set()
        worker=LatestOnlyWorker(work,result)
        try:
            worker.submit({"request_id":"a"})
            self.assertTrue(started.wait(2))
            worker.submit({"request_id":"b"})
            worker.submit({"request_id":"c"})
            release.set()
            self.assertTrue(done.wait(2))
            self.assertEqual(calls,["a","c"])
            self.assertEqual(worker.overwritten,1)
        finally:worker.close()

    def test_occlusion_and_rate_limit(self):
        trigger=QueryTrigger()
        status=dict(local_valid=True,global_valid=False,T_odom_base=np.eye(4).tolist())
        black=np.zeros((4,4,3),np.uint8)
        self.assertEqual(trigger.decide(black,0.,status),(False,"occluded_or_low_texture"))
        rgb=np.arange(48,dtype=np.uint8).reshape(4,4,3)*5
        self.assertTrue(trigger.decide(rgb,.1,status)[0])
        self.assertEqual(trigger.decide(rgb,.2,status),(False,"rate_limited"))
        self.assertTrue(trigger.decide(rgb,1.2,status)[0])

    def test_ekf_only_raw_velocity_and_gyro_config(self):
        import yaml
        path=Path(__file__).parents[1]/"configs/robot_sim/ekf.yaml"
        p=yaml.safe_load(path.read_text())["ekf_filter_node"]["ros__parameters"]
        self.assertEqual([i for i,v in enumerate(p["odom0_config"]) if v],[6,7,11])
        self.assertEqual([i for i,v in enumerate(p["imu0_config"]) if v],[11])
        self.assertEqual(p["world_frame"],"odom")
        self.assertEqual(p["frequency"],30.)
        self.assertNotIn("odom1",p)


if __name__=="__main__":unittest.main()
