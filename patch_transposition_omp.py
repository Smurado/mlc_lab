with open("week8/teir_evaluator.cpp", "r") as f:
    text = f.read()

# We need to make sure that parallel evaluation does not mix the active loops in shared state.
# Ah, active_loops is passed by value! Thus each thread gets its own copy.
# Wait, float* out = ... is accessed. Does Transposition have a race condition?
# Let's check transposition policy 'parallel'.
