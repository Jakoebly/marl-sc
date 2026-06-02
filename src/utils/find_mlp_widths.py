def mlp_params(D, hidden, O):
    dims = [D] + hidden + [O]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

# Measured from the [PARAMCOUNT] print:
D_a, P_a = 49,  68106    # actor: input dim, A4 target params
D_c, P_c = 196, 108929   # critic: input dim, A4 target params

def best_matches(D, O, target, n_layers=2, step=4, tol=0.05):
    grid = range(16, 1025, step)
    hits = []
    if n_layers == 1:
        for h1 in grid:
            hidden = [h1]
            p = mlp_params(D, hidden, O)
            err = abs(p - target) / target
            if err <= tol:
                hits.append((err, hidden, p))
    else:
        for h1 in grid:
            for h2 in grid:
                hidden = [h1, h2]
                p = mlp_params(D, hidden, O)
                err = abs(p - target) / target
                if err <= tol:
                    hits.append((err, hidden, p))
    hits.sort()
    return hits

print("ACTOR  (target", P_a, ")")
for err, hidden, p in best_matches(D_a, 5, P_a, n_layers=2)[:25]:
    print(f"  {hidden} -> {p} params ({err*100:+.2f}%)")

print("CRITIC (target", P_c, ")")
for err, hidden, p in best_matches(D_c, 1, P_c, n_layers=1, step=1)[:25]:
    print(f"  {hidden} -> {p} params ({err*100:+.2f}%)")