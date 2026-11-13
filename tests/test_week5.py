"""Week 5: cell subsets with fixed users, the week-5 grids and the comparison run definitions."""
import os

from experiments.week3_eval import CONTROLLERS, attacks
from experiments.week5_compare import ATTACK_KEYS, CONTROLLER_KEYS, controller_name, valid
from experiments.week5_launch import command, compare_runs, fed_dir, fl_runs
from rra.fed.cells import CELLS, SCALE_ORDER, cell_env_config, cell_subset


def test_cell_subset_prefixes_and_fixed_users():
    assert sorted(SCALE_ORDER) == list(range(len(CELLS)))
    ids4, specs4 = cell_subset(4, users=30)
    ids2, _ = cell_subset(2)
    assert ids2 == ids4[:2] and len(set(ids4)) == 4
    assert all(s.users == (30, 30) for s in specs4)
    assert [s.slice_mix for s in specs4] == [CELLS[i].slice_mix for i in ids4]
    assert cell_env_config(specs4[0]).target_users == (30, 30)
    assert cell_subset(1)[1][0] == CELLS[SCALE_ORDER[0]]


def test_week5_grids():
    scale = fl_runs("fl_scale", (1, 2, 4, 8), (10, 20, 30, 40), ["fedavg", "two_tier"], ["clean", "malicious"])
    assert sum(r["n"] == 1 for r in scale) == 4 and all(r["agg"] == "local" for r in scale if r["n"] == 1)
    assert len(scale) == 4 * (1 + 3 * (1 + 2 * 2))
    assert all(r["malicious"] <= 1 and r["deceived"] == 0 for r in scale)
    learn = fl_runs("fl_learn", (1, 2, 4, 8), (10, 20), ["fedavg", "two_tier"], ["clean", "malicious"])
    assert all(r["malicious"] == 0 and r["init"] == "scratch" for r in learn)
    r = next(x for x in scale if x["agg"] == "two_tier" and x["malicious"] == 1 and x["n"] == 4 and x["users"] == 30)
    cmd = command(r, 2, "results/week5", 6)
    assert cmd[cmd.index("--n-cells") + 1] == "4" and cmd[cmd.index("--users") + 1] == "30"
    # the launcher's directory name is the one experiments/week4_fed.py writes
    assert os.path.basename(fed_dir(r, 2, "results/week5")) == "two_tier_k0m1_n4_U30_s2"


def test_compare_definitions():
    names = {n for n, _, _ in attacks()}
    assert set(ATTACK_KEYS.values()) <= names
    assert {controller_name(k) for k in CONTROLLER_KEYS if k != "pf"} == {c[0] for c in CONTROLLERS}
    runs = compare_runs((10, 40), list(CONTROLLER_KEYS), list(ATTACK_KEYS))
    assert not any(r["controller"] == "pf" and r["attack"] in ("whitebox", "adaptive") for r in runs)
    assert len(runs) == 2 * (8 * 5 + 3) and valid("prop_b", "adaptive") and not valid("pf", "whitebox")
