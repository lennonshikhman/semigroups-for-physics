import matplotlib.pyplot as plt
import numpy as np


def save_sg_vs_rollout(df, out_pdf, out_png):
    plt.figure(figsize=(5, 4))
    plt.scatter(df['sg_unseen'], df['rollout_auc'], s=14, alpha=0.7)
    plt.xlabel('Unseen semigroup error')
    plt.ylabel('Rollout AUC error')
    plt.tight_layout(); plt.savefig(out_pdf); plt.savefig(out_png, dpi=150); plt.close()


def save_rollout_curves(curves_by_variant, out_pdf, out_png):
    plt.figure(figsize=(6, 4))
    for k, v in curves_by_variant.items():
        plt.plot(np.mean(v, axis=0), label=k)
    plt.xlabel('Step'); plt.ylabel('Relative L2'); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_pdf); plt.savefig(out_png, dpi=150); plt.close()


def save_seen_vs_unseen(df, out_pdf, out_png):
    plt.figure(figsize=(5, 4))
    plt.scatter(df['sg_seen'], df['sg_unseen'], s=14, alpha=0.7)
    m = max(df['sg_seen'].max(), df['sg_unseen'].max())
    plt.plot([0, m], [0, m], 'k--', lw=1)
    plt.xlabel('Seen SG error'); plt.ylabel('Unseen SG error')
    plt.tight_layout(); plt.savefig(out_pdf); plt.savefig(out_png, dpi=150); plt.close()


def save_lambda_ablation(df, out_pdf, out_png):
    plt.figure(figsize=(5, 4))
    g = df.groupby('lambda_sg')['sg_unseen'].mean().reset_index()
    plt.plot(g['lambda_sg'], g['sg_unseen'], marker='o')
    plt.xscale('log'); plt.xlabel('lambda_sg'); plt.ylabel('Unseen SG error')
    plt.tight_layout(); plt.savefig(out_pdf); plt.savefig(out_png, dpi=150); plt.close()
