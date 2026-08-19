from arox.core.background import BackgroundTaskBroker


def test_observing_completion_suppresses_notice():
    broker = BackgroundTaskBroker()
    broker.register("task")
    broker.complete("task", "finished")
    broker.observe("task")

    assert broker.drain_notices() == []


def test_completion_notice_is_delivered_once():
    broker = BackgroundTaskBroker()
    broker.register("task")
    broker.complete("task", "finished")

    assert broker.drain_notices() == ["finished"]
    assert broker.drain_notices() == []
