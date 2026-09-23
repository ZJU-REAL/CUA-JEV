import json

from cua_jev.cli import main


def test_tasks_lists_current_workflows_not_legacy_fixtures(capsys):
    assert main(["tasks"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in tasks} == {"edge", "excel", "vscode", "explorer"}
    assert next(item for item in tasks if item["name"] == "edge")["title"] == "Edge checkout workflow"
    assert all(item["steps"] > 0 for item in tasks)
