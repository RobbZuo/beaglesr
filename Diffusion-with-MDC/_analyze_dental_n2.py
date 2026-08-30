import json
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent
files = {
    "Bicubic": "dental_metrics_bicubic.txt",
    "SRGAN": "dental_metrics_srgan.txt",
    "SRDiff": "dental_metrics_srdiff.txt",
    "SwinIR": "dental_metrics_swinir.txt",
    "Ours w/ freq loss": "dental_metrics_our_withfreqloss.txt",
    "Ours w/ QR loss": "dental_metrics_our_withqrloss.txt",
    "Ours w/o MDC": "dental_metrics_our_withoutMDC.txt",
    "Ours w/ MDC": "dental_metrics_our_withMDC.txt",
}
order = list(files.keys())
metrics = ["PSNR", "SSIM", "LPIPS"]
N_SLICE = 256
N_SEQ = 10
N_SEQ_PER_SAMPLE = 5
N_SAMPLE = 2


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


# slice -> sequence means (10,) -> sample means (2,)
seq_means = {}  # method -> (10, 3)
sample_means = {}  # method -> (2, 3)
for name, fn in files.items():
    arr = load(base / fn)
    assert arr.shape == (2560, 3), (name, arr.shape)
    seq = arr.reshape(N_SEQ, N_SLICE, 3).mean(axis=1)  # (10, 3)
    samp = seq.reshape(N_SAMPLE, N_SEQ_PER_SAMPLE, 3).mean(axis=1)  # (2, 3)
    seq_means[name] = seq
    sample_means[name] = samp

# ---- intermediate: sequence means (N_seq=10) ----
seq_long = ["Method,Sequence,Sample,PSNR,SSIM,LPIPS"]
seq_txt = [
    "Dental metrics — per-sequence means (10 sequences × 256 slices)",
    "S1–S5 → Sample A; S6–S10 → Sample B",
    "=" * 80,
]
for name in order:
    seq = seq_means[name]
    seq_txt.append(f"\n[{name}]")
    seq_txt.append(f"{'Seq':<8}{'Sample':<10}{'PSNR':>12}{'SSIM':>12}{'LPIPS':>12}")
    seq_txt.append("-" * 54)
    for s in range(N_SEQ):
        samp_label = "A" if s < 5 else "B"
        seq_long.append(
            f"{name},Seq{s + 1},{samp_label},"
            f"{seq[s, 0]:.6f},{seq[s, 1]:.6f},{seq[s, 2]:.6f}"
        )
        seq_txt.append(
            f"{'Seq' + str(s + 1):<8}{samp_label:<10}"
            f"{seq[s, 0]:12.6f}{seq[s, 1]:12.6f}{seq[s, 2]:12.6f}"
        )
(base / "dental_sequence_means.csv").write_text("\n".join(seq_long) + "\n", encoding="utf-8")
(base / "dental_sequence_means.txt").write_text("\n".join(seq_txt) + "\n", encoding="utf-8")

for mi, mname in enumerate(metrics):
    header = ["Sequence", "Sample"] + order
    rows = [",".join(header)]
    for s in range(N_SEQ):
        samp_label = "A" if s < 5 else "B"
        vals = [f"Seq{s + 1}", samp_label] + [
            f"{seq_means[n][s, mi]:.6f}" for n in order
        ]
        rows.append(",".join(vals))
    (base / f"dental_sequence_means_{mname}.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

# ---- sample-level (N=2): Sample A / B ----
samp_long = ["Method,Sample,PSNR,SSIM,LPIPS"]
samp_txt = [
    "Dental metrics — per-sample means (N=2)",
    "Sample A = mean of Seq1–5; Sample B = mean of Seq6–10",
    "=" * 80,
]
for name in order:
    sm = sample_means[name]
    samp_txt.append(f"\n[{name}]")
    samp_txt.append(f"{'Sample':<10}{'PSNR':>12}{'SSIM':>12}{'LPIPS':>12}")
    samp_txt.append("-" * 46)
    for i, lab in enumerate(["A", "B"]):
        samp_long.append(
            f"{name},{lab},{sm[i, 0]:.6f},{sm[i, 1]:.6f},{sm[i, 2]:.6f}"
        )
        samp_txt.append(
            f"{lab:<10}{sm[i, 0]:12.6f}{sm[i, 1]:12.6f}{sm[i, 2]:12.6f}"
        )
    # overall = mean of 2 samples
    overall = sm.mean(axis=0)
    # range / half-range as spread (not SD with df=1 which is unstable for N=2)
    spread = np.abs(sm[0] - sm[1]) / 2  # mean absolute deviation from overall for N=2
    samp_txt.append("-" * 46)
    samp_txt.append(
        f"{'Mean':<10}{overall[0]:12.6f}{overall[1]:12.6f}{overall[2]:12.6f}"
    )
    samp_txt.append(
        f"{'|A-B|/2':<10}{spread[0]:12.6f}{spread[1]:12.6f}{spread[2]:12.6f}"
    )
(base / "dental_sample_means_N2.csv").write_text("\n".join(samp_long) + "\n", encoding="utf-8")
(base / "dental_sample_means_N2.txt").write_text("\n".join(samp_txt) + "\n", encoding="utf-8")

for mi, mname in enumerate(metrics):
    header = ["Sample"] + order
    rows = [",".join(header)]
    for i, lab in enumerate(["A", "B"]):
        vals = [lab] + [f"{sample_means[n][i, mi]:.6f}" for n in order]
        rows.append(",".join(vals))
    means = ["Mean"] + [f"{sample_means[n][:, mi].mean():.6f}" for n in order]
    spreads = ["|A-B|/2"] + [
        f"{abs(sample_means[n][0, mi] - sample_means[n][1, mi]) / 2:.6f}" for n in order
    ]
    rows.append(",".join(means))
    rows.append(",".join(spreads))
    (base / f"dental_sample_means_N2_{mname}.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

# ---- final descriptive summary (no significance tests) ----
summary_csv = [
    "Method,PSNR_A,PSNR_B,PSNR_mean,PSNR_half_range,"
    "SSIM_A,SSIM_B,SSIM_mean,SSIM_half_range,"
    "LPIPS_A,LPIPS_B,LPIPS_mean,LPIPS_half_range"
]
summary_txt = [
    "Dental metrics — final descriptive summary (N=2 samples)",
    "NOTE: N=2 is too small for meaningful significance testing",
    "      (Mann–Whitney / Kruskal–Wallis cannot reach p<0.05 with n1=n2=2).",
    "      Reported: Sample A/B means, overall mean, and half-range |A-B|/2 as spread.",
    "=" * 110,
    f"{'Method':<22}{'PSNR mean':>14}{'|A-B|/2':>10}"
    f"{'SSIM mean':>14}{'|A-B|/2':>10}"
    f"{'LPIPS mean':>14}{'|A-B|/2':>10}",
    "-" * 110,
]

detail_txt = [
    "",
    "Per-sample breakdown (Sample A / Sample B)",
    "=" * 90,
    f"{'Method':<22}{'PSNR A':>10}{'PSNR B':>10}{'SSIM A':>10}{'SSIM B':>10}{'LPIPS A':>10}{'LPIPS B':>10}",
    "-" * 90,
]

rank_note = []
for name in order:
    sm = sample_means[name]
    overall = sm.mean(axis=0)
    half = np.abs(sm[0] - sm[1]) / 2
    summary_csv.append(
        f"{name},"
        f"{sm[0, 0]:.6f},{sm[1, 0]:.6f},{overall[0]:.6f},{half[0]:.6f},"
        f"{sm[0, 1]:.6f},{sm[1, 1]:.6f},{overall[1]:.6f},{half[1]:.6f},"
        f"{sm[0, 2]:.6f},{sm[1, 2]:.6f},{overall[2]:.6f},{half[2]:.6f}"
    )
    summary_txt.append(
        f"{name:<22}"
        f"{overall[0]:14.4f}{half[0]:10.4f}"
        f"{overall[1]:14.4f}{half[1]:10.4f}"
        f"{overall[2]:14.4f}{half[2]:10.4f}"
    )
    detail_txt.append(
        f"{name:<22}"
        f"{sm[0, 0]:10.4f}{sm[1, 0]:10.4f}"
        f"{sm[0, 1]:10.4f}{sm[1, 1]:10.4f}"
        f"{sm[0, 2]:10.4f}{sm[1, 2]:10.4f}"
    )

# Ranking by overall mean (descriptive only)
rank_txt = ["", "Descriptive ranking by overall mean (N=2; not a significance test)", "=" * 70]
for mi, mname in enumerate(metrics):
    higher_better = mname != "LPIPS"
    scored = [(n, float(sample_means[n][:, mi].mean())) for n in order]
    scored.sort(key=lambda x: x[1], reverse=higher_better)
    direction = "↑ higher better" if higher_better else "↓ lower better"
    rank_txt.append(f"\n{mname} ({direction})")
    for i, (n, v) in enumerate(scored, 1):
        mark = "  << MDC" if n == "Ours w/ MDC" else ""
        rank_txt.append(f"  {i}. {n:<22} {v:.6f}{mark}")

# Delta vs MDC (descriptive)
ref = "Ours w/ MDC"
delta_csv = ["Metric,vs,MDC_mean,Other_mean,Delta"]
delta_txt = [
    "",
    "Descriptive difference vs Ours w/ MDC (Delta = MDC − other; no p-values)",
    "=" * 80,
    f"{'Metric':<8}{'vs':<22}{'Delta':>12}{'MDC':>12}{'Other':>12}",
    "-" * 80,
]
for mi, mname in enumerate(metrics):
    ref_m = float(sample_means[ref][:, mi].mean())
    for name in order:
        if name == ref:
            continue
        other_m = float(sample_means[name][:, mi].mean())
        d = ref_m - other_m
        delta_csv.append(f"{mname},{name},{ref_m:.6f},{other_m:.6f},{d:.6f}")
        delta_txt.append(
            f"{mname:<8}{name:<22}{d:+12.4f}{ref_m:12.4f}{other_m:12.4f}"
        )

(base / "dental_stats_summary_N2.csv").write_text(
    "\n".join(summary_csv) + "\n", encoding="utf-8"
)
(base / "dental_stats_delta_vs_MDC_N2.csv").write_text(
    "\n".join(delta_csv) + "\n", encoding="utf-8"
)

final_txt = (
    summary_txt
    + detail_txt
    + rank_txt
    + delta_txt
    + [
        "",
        "Notes:",
        "- 2560 slices → 10 sequences (256 each) → 2 samples (5 sequences each).",
        "- Significance tests are not reported: with N=2 per group, non-parametric",
        "  tests cannot produce p<0.05 regardless of effect size.",
        "- |A-B|/2 is a simple spread measure between the two samples (not a SD).",
    ]
)
(base / "dental_stats_final_N2.txt").write_text("\n".join(final_txt) + "\n", encoding="utf-8")

# JSON dump
out = {
    "note": "N=2; no significance tests. Sample A=Seq1-5, Sample B=Seq6-10.",
    "methods": {},
}
for name in order:
    sm = sample_means[name]
    overall = sm.mean(axis=0)
    half = np.abs(sm[0] - sm[1]) / 2
    out["methods"][name] = {
        "sample_A": {m: float(sm[0, i]) for i, m in enumerate(metrics)},
        "sample_B": {m: float(sm[1, i]) for i, m in enumerate(metrics)},
        "mean": {m: float(overall[i]) for i, m in enumerate(metrics)},
        "half_range": {m: float(half[i]) for i, m in enumerate(metrics)},
        "sequence_means": {
            m: [float(x) for x in seq_means[name][:, i]] for i, m in enumerate(metrics)
        },
    }
(base / "_stats_dental_n2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

print("=== Dental N=2 summary (mean | half-range) ===")
for name in order:
    sm = sample_means[name]
    overall = sm.mean(axis=0)
    half = np.abs(sm[0] - sm[1]) / 2
    print(f"\n{name}")
    print(f"  A: PSNR={sm[0,0]:.4f} SSIM={sm[0,1]:.4f} LPIPS={sm[0,2]:.4f}")
    print(f"  B: PSNR={sm[1,0]:.4f} SSIM={sm[1,1]:.4f} LPIPS={sm[1,2]:.4f}")
    print(
        f"  mean: {overall[0]:.4f}/{overall[1]:.4f}/{overall[2]:.4f}  "
        f"half-range: {half[0]:.4f}/{half[1]:.4f}/{half[2]:.4f}"
    )

print("\nSaved:")
print("  dental_sequence_means.csv / .txt / _{PSNR,SSIM,LPIPS}.csv")
print("  dental_sample_means_N2.csv / .txt / _{PSNR,SSIM,LPIPS}.csv")
print("  dental_stats_summary_N2.csv")
print("  dental_stats_delta_vs_MDC_N2.csv")
print("  dental_stats_final_N2.txt")
