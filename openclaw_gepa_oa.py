import gepa.optimize_anything as oa

def evaluate(candidate: str) -> float:
    score, diagnostic = run_my_system(candidate)
    oa.log(f"Error: {diagnostic}")  # captured as ASI
    return score

# Start from an existing artifact…
result = oa.optimize_anything(
    seed_candidate="<your initial artifact>",
    evaluator=evaluate,
)

# … or just describe what you need.
result = oa.optimize_anything(
    evaluator=evaluate,
    objective="Generate a Python function `reverse()` that reverses a string.",
    config=oa.GEPAConfig(engine=oa.EngineConfig(max_metric_calls=100)),
)

print(result.best_candidate)