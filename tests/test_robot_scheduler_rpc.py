"""Real lightweight child-process protocol tests; never fake localization evidence."""
import json
import sys

import pytest

from research3d.robot_sim.scheduler import AceRpcProcess


def test_ready_handshake_consumes_only_control_receipt(tmp_path):
    program = """import json,sys
print(json.dumps({'event':'ready','map_id':'test','map_sha256':'sha'}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    print(json.dumps({'request_id':request['request_id'],'protocol_echo_only':True}),flush=True)
"""
    rpc=AceRpcProcess([sys.executable,"-u","-c",program],tmp_path/"worker.log",timeout_s=3.)
    try:
        assert rpc.wait_ready()["map_id"]=="test"
        assert rpc.wait_ready()["event"]=="ready"
        assert rpc({"request_id":"one"})=={"request_id":"one","protocol_echo_only":True}
        assert rpc({"request_id":"two"})["request_id"]=="two"
    finally:rpc.close()
    assert rpc.process.poll() is not None


def test_ready_fatal_preserves_real_error_and_cleans_own_child(tmp_path):
    program="import json; print(json.dumps({'event':'fatal','error_type':'MissingMap','error':'map artifact absent'}),flush=True)"
    rpc=AceRpcProcess([sys.executable,"-u","-c",program],tmp_path/"worker.log",timeout_s=3.)
    with pytest.raises(RuntimeError,match="MissingMap: map artifact absent"):
        rpc.wait_ready()
    assert rpc.process.poll() is not None


def test_ready_timeout_is_bounded_and_terminates_child(tmp_path):
    rpc=AceRpcProcess([sys.executable,"-u","-c","import time; time.sleep(30)"],tmp_path/"worker.log",timeout_s=.1)
    with pytest.raises(TimeoutError):rpc.wait_ready()
    assert rpc.process.poll() is not None
