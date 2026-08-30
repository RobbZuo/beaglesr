import json
import numpy as np
from pathlib import Path
from scipy import stats

base = Path(__file__).resolve().parent
files = {
    "Bicubic": "5T_metrics_bicubic.txt",
    "SRGAN": "5T_metrics_srgan.txt",
    "SRDiff": "5T_metrics_srdiff.txt",
    "SwinIR": "5T_metrics_swinir.txt",
    "Ours w/ freq loss": "5T_metrics_our_withfreqloss.txt",
    "Ours w/ QR loss": "5T_metrics_our_withqrloss.txt",
    "Ours w/o MDC": "5T_metrics_our_withoutMDC.txt",
    "Ours w/ MDC": "5T_metrics_our_withMDC.txt",
}
order = list(files.keys())
metrics = ["PSNR", "SSIM", "LPIPS"]
REF = "Ours w/ MDC"
N_SLICE = 260
N_SAMPLE = 10


def load(path: Path) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("—") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(rows)


# slice-level -> sample means (N=10, paired across methods)
sample_means = {}
for name, fn in files.items():
    arr = load(base / fn)
    assert arr.shape == (N_SAMPLE * N_SLICE, 3), (name, arr.shape)
    sample_means[name] = arr.reshape(N_SAMPLE, N_SLICE, 3).mean(axis=1)

# ---- descriptive summary ----
summary = []
for name in order:
    sm = sample_means[name]
    row = {"method": name}
    for i, m in enumerate(metrics):
        v = sm[:, i]
        row[m] = {
            "mean": round(float(v.mean()), 6),
            "std": round(float(v.std(ddof=1)), 6),
            "median": round(float(np.median(v)), 6),
            "min": round(float(v.min()), 6),
            "max": round(float(v.max()), 6),
            "values": [round(float(x), 6) for x in v],
        }
    summary.append(row)

# ---- Friedman (paired omnibus, 8 methods x 10 samples) ----
friedman = []
for i, m in enumerate(metrics):
    groups = [sample_means[n][:, i] for n in order]
    stat, p = stats.friedmanchisquare(*groups)
    friedman.append({"metric": m, "chi2": round(float(stat), 4), "p": float(p)})

# ---- Wilcoxon signed-rank: REF vs each other (paired, Bonferroni x7) ----
n_comp = len(order) - 1
pairwise = []
for i, m in enumerate(metrics):
    ref_vals = sample_means[REF][:, i]
    for name in order:
        if name == REF:
            continue
        other = sample_means[name][:, i]
        stat, p = stats.wilcoxon(ref_vals, other, alternative="two-sided")
        if m == "LPIPS":
            _, p_better = stats.wilcoxon(ref_vals, other, alternative="less")
        else:
            _, p_better = stats.wilcoxon(ref_vals, other, alternative="greater")
        p_bonf = min(float(p) * n_comp, 1.0)
        pairwise.append(
            {
                "metric": m,
                "vs": name,
                "ref_mean": round(float(ref_vals.mean()), 6),
                "other_mean": round(float(other.mean()), 6),
                "delta": round(float(ref_vals.mean() - other.mean()), 6),
                "statistic": float(stat),
                "p": float(p),
                "p_bonf": p_bonf,
                "p_better": float(p_better),
                "sig": bool(p_bonf < 0.05),
            }
        )

out = {
    "design": "N=10 paired sample means (260 slices/sample); Friedman + Wilcoxon",
    "summary": summary,
    "friedman": friedman,
    "pairwise": pairwise,
}
(base / "_stats_5t_n10.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

# ---- per-sample intermediate tables ----
long_lines = ["Method,Sample,PSNR,SSIM,LPIPS"]
for name in order:
    sm = sample_means[name]
    for s in range(N_SAMPLE):
        long_lines.append(
            f"{name},S{s + 1},{sm[s, 0]:.6f},{sm[s, 1]:.6f},{sm[s, 2]:.6f}"
        )
(base / "5T_sample_means_N10.csv").write_text("\n".join(long_lines) + "\n", encoding="utf-8")

for mi, mname in enumerate(metrics):
    header = ["Sample"] + order
    rows = [",".join(header)]
    for s in range(N_SAMPLE):
        vals = [f"S{s + 1}"] + [f"{sample_means[n][s, mi]:.6f}" for n in order]
        rows.append(",".join(vals))
    rows.append(",".join(["Mean"] + [f"{sample_means[n][:, mi].mean():.6f}" for n in order]))
    rows.append(",".join(["SD"] + [f"{sample_means[n][:, mi].std(ddof=1):.6f}" for n in order]))
    (base / f"5T_sample_means_N10_{mname}.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

txt = [
    "5T metrics — per-sample means (N=10, 260 slices/sample, paired across methods)",
    "=" * 100,
]
for name in order:
    sm = sample_means[name]
    txt.append(f"\n[{name}]")
    txt.append(f"{'Sample':<8}{'PSNR':>12}{'SSIM':>12}{'LPIPS':>12}")
    txt.append("-" * 44)
    for s in range(N_SAMPLE):
        txt.append(
            f"{'S' + str(s + 1):<8}{sm[s, 0]:12.6f}{sm[s, 1]:12.6f}{sm[s, 2]:12.6f}"
        )
    txt.append("-" * 44)
    txt.append(
        f"{'Mean':<8}{sm[:, 0].mean():12.6f}{sm[:, 1].mean():12.6f}{sm[:, 2].mean():12.6f}"
    )
    txt.append(
        f"{'SD':<8}{sm[:, 0].std(ddof=1):12.6f}{sm[:, 1].std(ddof=1):12.6f}{sm[:, 2].std(ddof=1):12.6f}"
    )
(base / "5T_sample_means_N10.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

# ---- final stats tables ----
desc_csv = ["Method,PSNR_mean,PSNR_SD,SSIM_mean,SSIM_SD,LPIPS_mean,LPIPS_SD"]
desc_txt = [
    "Table 1. Descriptive statistics (mean ± SD, N=10 paired sample means)",
    "=" * 90,
    f"{'Method':<22}{'PSNR':>18}{'SSIM':>18}{'LPIPS':>18}",
    "-" * 90,
]
for row in summary:
    name = row["method"]
    p, s, l = row["PSNR"], row["SSIM"], row["LPIPS"]
    desc_csv.append(
        f"{name},{p['mean']:.6f},{p['std']:.6f},"
        f"{s['mean']:.6f},{s['std']:.6f},"
        f"{l['mean']:.6f},{l['std']:.6f}"
    )
    desc_txt.append(
        f"{name:<22}"
        f"{p['mean']:.2f} ± {p['std']:.3f}".rjust(18)
        + f"{s['mean']:.4f} ± {s['std']:.4f}".rjust(18)
        + f"{l['mean']:.4f} ± {l['std']:.4f}".rjust(18)
    )
(base / "5T_stats_summary_N10.csv").write_text("\n".join(desc_csv) + "\n", encoding="utf-8")

fr_csv = ["Metric,chi2,p,significant_alpha0.05"]
fr_txt = [
    "",
    "Table 2. Friedman test across 8 methods (N=10 paired samples)",
    "=" * 60,
    f"{'Metric':<10}{'chi2':>12}{'p':>16}{'Significant':>14}",
    "-" * 60,
]
for k in friedman:
    sig = "Yes" if k["p"] < 0.05 else "No"
    fr_csv.append(f"{k['metric']},{k['chi2']:.4f},{k['p']:.6e},{sig}")
    fr_txt.append(f"{k['metric']:<10}{k['chi2']:12.4f}{k['p']:16.6e}{sig:>14}")
(base / "5T_stats_friedman_N10.csv").write_text("\n".join(fr_csv) + "\n", encoding="utf-8")

wx_csv = ["Metric,vs,MDC_mean,Other_mean,Delta,W,p,p_bonferroni,significant"]
wx_txt = [
    "",
    f"Table 3. Wilcoxon signed-rank: {REF} vs others (paired, Bonferroni ×7, α=0.05)",
    "Delta = mean(MDC) − mean(other)",
    "=" * 110,
    f"{'Metric':<8}{'vs':<22}{'Delta':>10}{'p':>12}{'p_bonf':>12}{'Sig':>6}",
    "-" * 110,
]
for x in pairwise:
    sig = "Yes" if x["sig"] else "No"
    wx_csv.append(
        f"{x['metric']},{x['vs']},{x['ref_mean']:.6f},{x['other_mean']:.6f},"
        f"{x['delta']:.6f},{x['statistic']:.1f},{x['p']:.6e},{x['p_bonf']:.6e},{sig}"
    )
    wx_txt.append(
        f"{x['metric']:<8}{x['vs']:<22}{x['delta']:+10.4f}"
        f"{x['p']:12.4e}{x['p_bonf']:12.4e}{sig:>6}"
    )
(base / "5T_stats_wilcoxon_vs_MDC_N10.csv").write_text(
    "\n".join(wx_csv) + "\n", encoding="utf-8"
)

# keep legacy filename as symlink-like copy for compatibility
(base / "5T_stats_mannwhitney_vs_MDC_N10.csv").write_text(
    (base / "5T_stats_wilcoxon_vs_MDC_N10.csv").read_text(encoding="utf-8"),
    encoding="utf-8",
)
(base / "5T_stats_kruskal_N10.csv").write_text(
    (base / "5T_stats_friedman_N10.csv").read_text(encoding="utf-8"),
    encoding="utf-8",
)

final_txt = desc_txt + fr_txt + wx_txt + [
    "",
    "Notes:",
    "- Analysis unit: N=10 sample means (260 consecutive slices averaged per sample).",
    "- All methods evaluated on the same 10 samples → paired non-parametric tests.",
    "- Friedman: omnibus test across 8 methods on paired observations.",
    "- Wilcoxon signed-rank: pairwise two-sided; Bonferroni correction = p × 7.",
]
(base / "5T_stats_final_N10.txt").write_text("\n".join(final_txt) + "\n", encoding="utf-8")

print("=== Summary (mean ± SD, N=10 paired) ===")
for row in summary:
    print(row["method"])
    for m in metrics:
        s = row[m]
        print(f"  {m}: {s['mean']:.4f} ± {s['std']:.4f}")

print("\n=== Friedman (8 methods, paired) ===")
for k in friedman:
    print(f"{k['metric']}: chi2={k['chi2']}, p={k['p']:.3e}")

print(f"\n=== Wilcoxon vs {REF} (Bonferroni ×{n_comp}) ===")
for m in metrics:
    print(f"\n{m}:")
    for x in pairwise:
        if x["metric"] != m:
            continue
        mark = "*" if x["sig"] else "ns"
        print(
            f"  vs {x['vs']}: Δ={x['delta']:+.4f}  p={x['p']:.4f}  "
            f"p_bonf={x['p_bonf']:.4f}  [{mark}]"
        )

print("\nSaved:")
print("  5T_sample_means_N10.*")
print("  5T_stats_summary_N10.csv")
print("  5T_stats_friedman_N10.csv")
print("  5T_stats_wilcoxon_vs_MDC_N10.csv")
print("  5T_stats_final_N10.txt")
