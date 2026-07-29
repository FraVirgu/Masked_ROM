import os

from CCO_DOMAIN import resolve_graph_paths


def test_resolve_graph_paths_prefers_existing_liver_toy():
    graph_dir, obj_path = resolve_graph_paths("graphExport", "graphExport/domain.obj")

    assert graph_dir == os.path.join("graph", "liver_toy")
    assert obj_path == os.path.join("graph", "liver_toy", "domain.obj")
