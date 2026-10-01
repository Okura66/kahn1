"""Round-3 brief inputs: disjoint name pools, mechanisms that fit each document's family."""

from training.teacher_catalog import MECHANISMS, assign_mechanisms, author_names


def test_name_pools_are_disjoint_and_skip_recurring_names():
    existing = ["Mira Holmberg met Mira Okonkwo.", "Mira Sato and the Altamar Bank.", "Altamar Logistics"]
    pools = author_names(6, ["en", "fr"] * 3, existing, people=10, orgs=3)
    people = [p for pool in pools for p in pool["people"]]
    orgs = [o for pool in pools for o in pool["organisations"]]
    assert len(people) == len(set(people)) == 60 and len(orgs) == len(set(orgs)) == 18
    assert not any(p.startswith("Mira ") for p in people)          # in 2+ existing documents
    assert not any(o.startswith("Altamar ") for o in orgs)
    for pool in pools:                                             # no first or last name twice per author
        assert len({p.split()[0] for p in pool["people"]}) == 10
        assert len({p.split()[-1] for p in pool["people"]}) == 10


def test_mechanisms_follow_the_trio_and_spread():
    trios = [("long_policy", "judge", "temporal_numeric")] * 5
    plans = assign_mechanisms(trios)
    for plan in plans:
        assert len(plan) == 10 and len(set(plan)) == 10
        for k, m in enumerate(plan):
            assert m in MECHANISMS[trios[0][k % 3]]
    uses = [sum(m in p for p in plans) for m in MECHANISMS["judge"]]
    assert max(uses) - min(uses) <= 1                              # used as evenly as possible
