from reindexrepair.smoke import run_smoke


def test_smoke_scenarios_have_expected_control_behavior():
    payload = run_smoke(incidents_per_scenario=2)
    assert payload["paper_evidence"] is False
    assert payload["by_scenario"]["rank_substitute"]["f2_safe"] == 0
    assert payload["by_scenario"]["boundary_migration"]["f2_safe"] == 0
    assert payload["by_scenario"]["no_mirage_control"]["f2_safe"] == 2

