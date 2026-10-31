from __future__ import annotations

from issueforge_server.models import IngestJob, RepoGroup
from issueforge_server.settings_store import get_raw
from issueforge_server.worker import Worker


def _enabled_group(name: str = "跟踪组") -> RepoGroup:
    return RepoGroup(
        name=name,
        description="test",
        criteria="CUDA",
        threshold=0.5,
        tracking_enabled=True,
    )


def test_worker_schedules_tracking_for_enabled_groups(session, factory, box, env) -> None:
    group = _enabled_group()
    session.add(group)
    session.commit()
    group_id = group.id
    session.close()

    worker = Worker(factory, box, env)
    worker.run_once()

    session = factory()
    try:
        job = session.query(IngestJob).filter_by(group_id=group_id).one()
        assert job.kind == "track"
        assert job.status == "queued"
        assert get_raw(session, "worker_heartbeat")
        assert get_raw(session, "worker_error") == ""
    finally:
        session.close()


def test_worker_does_not_duplicate_track_jobs_within_interval(session, factory, box, env) -> None:
    group = _enabled_group()
    session.add(group)
    session.commit()
    group_id = group.id
    session.close()

    worker = Worker(factory, box, env)
    worker.run_once()
    worker.run_once()

    session = factory()
    try:
        jobs = session.query(IngestJob).filter_by(group_id=group_id).all()
        assert len(jobs) == 1
    finally:
        session.close()


def test_worker_marks_missing_group_job_failed(session, factory, box, env) -> None:
    session.add(IngestJob(kind="track", group_id=9999, payload_json="{}"))
    session.commit()
    session.close()

    worker = Worker(factory, box, env)
    worker.run_once()

    session = factory()
    try:
        job = session.query(IngestJob).filter_by(group_id=9999).one()
        assert job.status == "failed"
        assert "仓库组不存在" in job.error
    finally:
        session.close()
