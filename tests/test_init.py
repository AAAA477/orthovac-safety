import orthovac


def test_every_public_name_is_importable():
    missing = [n for n in orthovac.__all__ if not hasattr(orthovac, n)]
    assert not missing, missing


def test_the_spec_interface_is_present():
    for name in ('setup', 'show_plan', 'build_adapters', 'upload_adapters', 'run_evals', 'merge_summaries',
                 'sync_status', 'line_graphs', 'find_results'):
        assert callable(getattr(orthovac, name)), name
