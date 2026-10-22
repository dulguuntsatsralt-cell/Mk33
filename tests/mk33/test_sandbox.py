"""LocalSandbox on Linux, macOS and Windows (CI runs this file on all three)."""

import sys
import time

import pytest

from mk33.scaffold.sandbox import IS_WINDOWS, LocalSandbox, find_windows_bash

PY = f'"{sys.executable}"'


@pytest.fixture
def sbx(tmp_path):
    s = LocalSandbox(tmp_path)
    yield s
    s.kill_all()


def test_exec_returns_output_and_exit_code(sbx):
    r = sbx.exec(f"{PY} -c \"print('hi'); import sys; sys.exit(3)\"", timeout_s=30)
    assert r.returncode == 3
    assert "hi" in r.output
    assert not r.timed_out


def test_exec_runs_in_workspace(sbx, tmp_path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    r = sbx.exec(f"{PY} -c \"import os; print(sorted(os.listdir('.')))\"", timeout_s=30)
    assert "marker.txt" in r.output


def test_exec_handles_non_ascii_output(sbx):
    r = sbx.exec(f"{PY} -c \"print('Сайн байна уу 你好')\"", timeout_s=30)
    assert r.returncode == 0
    assert "Сайн байна уу 你好" in r.output


def test_timeout_kills_child_processes(sbx, tmp_path):
    # The shell starts a Python child that would write a file after 5 s. A timeout must
    # stop the child too, not only the shell, so the file never appears.
    flag = tmp_path / "late.txt"
    code = f"import time; time.sleep(5); open(r'{flag}', 'w').write('x')"
    t0 = time.time()
    r = sbx.exec(f'{PY} -c "{code}"', timeout_s=1)
    assert r.timed_out
    assert time.time() - t0 < 10
    time.sleep(6)
    assert not flag.exists()


def test_background_job_and_wait(sbx):
    jid = sbx.start_job(f"{PY} -c \"print('step 1'); print('ünïcode ok')\"")
    status, log, timed_out = sbx.wait_job(jid, timeout_s=30)
    assert status == "exited 0"
    assert "step 1" in log and "ünïcode ok" in log
    assert not timed_out
    # A second wait returns only new log text.
    assert sbx.wait_job(jid, timeout_s=1)[1] == ""


def test_kill_all_stops_running_jobs(sbx):
    jid = sbx.start_job(f"{PY} -c \"import time; time.sleep(60)\"")
    status, _, timed_out = sbx.wait_job(jid, timeout_s=1)
    assert status == "running" and timed_out
    sbx.kill_all()
    assert sbx._procs[jid].poll() is not None


def test_shell_name(sbx):
    if IS_WINDOWS:
        assert sbx.shell_name == ("bash" if find_windows_bash() else "cmd")
    else:
        assert sbx.shell_name == "bash"
