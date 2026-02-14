#!/usr/bin/env python3
"""
11_fig6_framework.py — Figure 6: Conceptual Framework
=======================================================

Creates a schematic diagram illustrating the policy implications of
recovery windows and intervention timing. This figure has no data
dependency — it is entirely programmatic.

Inputs
------
None (schematic diagram)

Outputs
-------
- outputs/figures/Figure6_conceptual_framework.png
- outputs/figures/Figure6_conceptual_framework.pdf
"""

import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PACKAGE_ROOT, 'outputs', 'figures')

# ---------------------------------------------------------------------------
# Nature-style formatting
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})


def main():
    """Generate Figure 6: Conceptual framework schematic."""
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Creating Figure 6: Conceptual framework...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, 'Recovery Windows and Intervention Timing',
            ha='center', fontsize=12, fontweight='bold')

    # ------------------------------------------------------------------
    # Left box: Early intervention
    # ------------------------------------------------------------------
    left_box = mpatches.FancyBboxPatch(
        (0.5, 2), 4, 3.5, boxstyle="round,pad=0.1",
        facecolor='#E8F4E8', edgecolor='#228833', linewidth=2)
    ax.add_patch(left_box)

    ax.text(2.5, 5.2, 'Early Intervention', ha='center', fontsize=10,
            fontweight='bold', color='#228833')
    ax.text(2.5, 4.5, '\u2022 High recovery probability (18%)',
            ha='center', fontsize=8)
    ax.text(2.5, 4.0, '\u2022 Assets still intact',
            ha='center', fontsize=8)
    ax.text(2.5, 3.5, '\u2022 Social networks functional',
            ha='center', fontsize=8)
    ax.text(2.5, 3.0, '\u2022 Aligns with natural resilience',
            ha='center', fontsize=8)
    ax.text(2.5, 2.3, '\u2192 Supports recovery', ha='center', fontsize=9,
            fontweight='bold', color='#228833')

    # ------------------------------------------------------------------
    # Right box: Delayed intervention
    # ------------------------------------------------------------------
    right_box = mpatches.FancyBboxPatch(
        (5.5, 2), 4, 3.5, boxstyle="round,pad=0.1",
        facecolor='#FDEAEA', edgecolor='#EE6677', linewidth=2)
    ax.add_patch(right_box)

    ax.text(7.5, 5.2, 'Delayed Intervention', ha='center', fontsize=10,
            fontweight='bold', color='#EE6677')
    ax.text(7.5, 4.5, '\u2022 Low recovery probability (2%)',
            ha='center', fontsize=8)
    ax.text(7.5, 4.0, '\u2022 Assets depleted',
            ha='center', fontsize=8)
    ax.text(7.5, 3.5, '\u2022 Livelihood erosion',
            ha='center', fontsize=8)
    ax.text(7.5, 3.0, '\u2022 System capacity degraded',
            ha='center', fontsize=8)
    ax.text(7.5, 2.3, '\u2192 Must create recovery', ha='center', fontsize=9,
            fontweight='bold', color='#EE6677')

    # Arrow between boxes
    ax.annotate('', xy=(5.3, 3.75), xytext=(4.7, 3.75),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(5, 4.1, 'Time', ha='center', fontsize=8, color='gray')

    # ------------------------------------------------------------------
    # Bottom: Policy implications by crisis type
    # ------------------------------------------------------------------
    ax.text(5, 1.5, 'Policy Implications by Crisis Type',
            ha='center', fontsize=10, fontweight='bold')

    types = [
        ('Seasonal\nspikes', '#4477AA', 'Anticipatory\naction'),
        ('Rapid\nonset', '#228833', 'Emergency\nresponse'),
        ('Protracted', '#EE6677', 'Sustained\ncommitment'),
    ]

    for i, (label, color, action) in enumerate(types):
        x = 1.5 + i * 3
        box = mpatches.FancyBboxPatch(
            (x - 0.8, 0.3), 2.2, 1, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.3)
        ax.add_patch(box)
        ax.text(x + 0.3, 0.95, label, ha='center', fontsize=8,
                fontweight='bold')
        ax.text(x + 0.3, 0.5, action, ha='center', fontsize=7)

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    png_path = os.path.join(FIG_DIR, 'Figure6_conceptual_framework.png')
    pdf_path = os.path.join(FIG_DIR, 'Figure6_conceptual_framework.pdf')
    plt.savefig(png_path, dpi=300, facecolor='white')
    plt.savefig(pdf_path, facecolor='white')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    main()
