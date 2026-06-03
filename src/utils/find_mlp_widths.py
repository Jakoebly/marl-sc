def mlp_params(D, hidden, O):
    dims = [D] + list(hidden) + [O]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

# Measured from the [PARAMCOUNT] print:
D_a, P_a = 55,  68298    # actor: input dim, A4 target params
D_c, P_c = 220, 109249   # critic: input dim, A4 target params

# Hidden-layer count per network. Set to mirror the E1-tuned topology of the env.
N_LAYERS_ACTOR  = 1
N_LAYERS_CRITIC = 1

def best_matches(D, O, target, n_layers, lo=16, hi=1400, step=None, tol=0.05):
    # Fine resolution for a 1-layer scan (cheap), coarser for the 2-layer grid.
    if step is None:
        step = 1 if n_layers == 1 else 4
    grid = range(lo, hi + 1, step)
    hits = []
    if n_layers == 1:
        for h1 in grid:
            hidden = [h1]
            p = mlp_params(D, hidden, O)
            err = abs(p - target) / target
            if err <= tol:
                hits.append((err, hidden, p))
    elif n_layers == 2:
        for h1 in grid:
            for h2 in grid:
                hidden = [h1, h2]
                p = mlp_params(D, hidden, O)
                err = abs(p - target) / target
                if err <= tol:
                    hits.append((err, hidden, p))
    else:
        raise ValueError(f"n_layers must be 1 or 2, got {n_layers}")
    hits.sort()
    return hits

print(f"ACTOR  (target {P_a}, {N_LAYERS_ACTOR} hidden layer(s))")
for err, hidden, p in best_matches(D_a, 5, P_a, N_LAYERS_ACTOR)[:50]:
    print(f"  {hidden} -> {p} params ({err*100:+.2f}%)")

print(f"CRITIC (target {P_c}, {N_LAYERS_CRITIC} hidden layer(s))")
for err, hidden, p in best_matches(D_c, 1, P_c, N_LAYERS_CRITIC)[:50]:
    print(f"  {hidden} -> {p} params ({err*100:+.2f}%)")
