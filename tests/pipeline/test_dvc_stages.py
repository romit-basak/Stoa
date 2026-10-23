import ast
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def dag_tasks() -> set[str]:
    """Names of the functions decorated with @task.bash in the DAG file, read without Airflow."""
    tree = ast.parse((REPO / "dags" / "stoa_pipeline.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for d in node.decorator_list:
                target = d.func if isinstance(d, ast.Call) else d
                if ast.unparse(target) == "task.bash":
                    names.add(node.name)
    return names


def test_every_dag_task_has_a_dvc_stage():
    # dvc_push is the versioning step itself, not a stage (dvc.yaml, header comment).
    stages = set(yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"])
    tasks = dag_tasks() - {"dvc_push"}
    assert tasks, "no tasks found in the DAG file"
    assert tasks == stages
