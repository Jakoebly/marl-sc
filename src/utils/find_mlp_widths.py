def mlp_params(D, hidden, O):
    dims = [D] + hidden + [O]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

# Measured from the [PARAMCOUNT] print:
D_a, P_a = 65,  68426    # actor: input dim, A4 target params
D_c, P_c = 260, 109377   # critic: input dim, A4 target params

def best_matches(D, O, target, tol=0.05):
    hits = []
    for h1 in range(16, 1025, 4):
        for h2 in range(16, 1025, 4):
            p = mlp_params(D, [h1, h2], O)
            err = abs(p - target) / target
            if err <= tol:
                hits.append((err, h1, h2, p))
    hits.sort()
    return hits

print("ACTOR  (target", P_a, ")")
for err, h1, h2, p in best_matches(D_a, 5, P_a)[:25]:
    print(f"  [{h1}, {h2}] -> {p} params ({err*100:+.2f}%)")

print("CRITIC (target", P_c, ")")
for err, h1, h2, p in best_matches(D_c, 1, P_c)[:25]:
    print(f"  [{h1}, {h2}] -> {p} params ({err*100:+.2f}%)")