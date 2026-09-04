from aster.candidates import CandidateCollector, CandidateSpec
from aster.integration import PsqlExplainRunner

QUERY = """
SELECT c.segment, count(*), sum(o.amount)
FROM customers c JOIN orders o ON o.customer_id = c.id
WHERE c.region = 'west' AND o.status IN (1,2,3)
GROUP BY c.segment
"""

runner = PsqlExplainRunner("postgresql://aster:aster@localhost:5432/aster")
collector = CandidateCollector(runner)
candidates = [
    CandidateSpec("native", {}),
    CandidateSpec("hash", {"enable_mergejoin": "off", "enable_nestloop": "off"}),
    CandidateSpec("merge", {"enable_hashjoin": "off", "enable_nestloop": "off"}),
    CandidateSpec("nested", {"enable_hashjoin": "off", "enable_mergejoin": "off"}),
]
discovered = collector.discover(QUERY, candidates)
assert len(discovered) >= 2, f"expected >=2 physical plans, got {len(discovered)}"
obs = collector.measure(
    QUERY,
    discovered[0],
    workload="ci-smoke",
    query_id="join-q1",
    query_template="join-aggregate",
    parameter_key="west-123",
    dataset_version="ci-v1",
    run_seed=7,
    code_revision="ci",
    warmups=0,
    repetitions=1,
)
assert obs[0].execution_time_ms > 0
print({"unique_plans": len(discovered), "execution_ms": obs[0].execution_time_ms})
