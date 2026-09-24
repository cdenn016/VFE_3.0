r"""Summarize precision_phase0.py JSON outputs: worst case over every measured batch and layer."""
import json
import sys


def main() -> None:
    for path in sys.argv[1:]:
        r = json.load(open(path, encoding="utf-8"))
        calls = r["calls"]
        print(f"== {r['run_dir']}  shape={r['shape']}  calls={len(calls)}  tau={calls[0]['tau']:.3f}  "
              f"kl_max={calls[0]['kl_max']}")
        triggers = {}
        for c in calls:
            for k, v in c["trigger_fraction"].items():
                v = max(v) if isinstance(v, list) else float(v)
                triggers[k] = max(triggers.get(k, 0.), v)
        print("  max trigger fraction over calls (either role):")
        for k, v in triggers.items():
            print(f"    {k:40s} {v:.4g}")
        for name in ("X_query", "X_key", "G_query", "U_block", "Sigma_token", "R_hager_1norm_est_query",
                     "X_whitened_query"):
            p50 = max(c["kappa"][name]["p50"] for c in calls)
            p99 = max(c["kappa"][name]["p99"] for c in calls)
            mx = max(c["kappa"][name]["max"] for c in calls)
            print(f"  kappa {name:26s} worst-call p50={p50:.3g} p99={p99:.3g} max={mx:.3g}")
        print("  S_q cond per head (range):",
              f"{min(min(c['kappa']['S_query_per_head']) for c in calls):.2f}-"
              f"{max(max(c['kappa']['S_query_per_head']) for c in calls):.2f}",
              " S_k:", f"{min(min(c['kappa']['S_key_per_head']) for c in calls):.2f}-"
              f"{max(max(c['kappa']['S_key_per_head']) for c in calls):.2f}")
        for name in calls[0]["whitened_factor_err"]:
            mx = max(c["whitened_factor_err"][name]["max"] for c in calls)
            print(f"  whitened factor err {name:26s} max={mx:.3g}")
        print("  KL error / tau on the causal support after the kl_max clamp (worst over calls):")
        for name in calls[0]["errors"]:
            rows = [c["errors"][name] for c in calls]
            if any("error" in e for e in rows):
                print(f"    {name:34s} RAISED: {[e['error'] for e in rows if 'error' in e][0][:80]}")
                continue
            mx = max(e["kl_err_over_tau"]["max"] for e in rows)
            p999 = max(e["kl_err_over_tau"]["p99.9"] for e in rows)
            beta = max(e["beta_err_max"] for e in rows)
            rec = sum(e.get("recovered_entries", 0) or 0 for e in rows)
            print(f"    {name:34s} max={mx:.2e} p99.9={p999:.2e} beta_max={beta:.2e} recovered={rec}")
        mx = max(c["pair_cancellation_ratio"]["max"] for c in calls)
        p99 = max(c["pair_cancellation_ratio"]["p99"] for c in calls)
        print(f"  pair cancellation sum|PG|/tr(PG): worst p99={p99:.3g} max={mx:.3g}")
        mx = max(c["pair_error_model_ratio_ws1"]["max"] for c in calls)
        print(f"  measured |dKL| / (eps (k_q + k_k) S): max={mx:.3g}")
        mx = max(c["pair_S_over_tau"]["max"] for c in calls)
        p50 = max(c["pair_S_over_tau"]["p50"] for c in calls)
        print(f"  S/tau on support: worst p50={p50:.3g} max={mx:.3g}")


if __name__ == "__main__":
    main()
