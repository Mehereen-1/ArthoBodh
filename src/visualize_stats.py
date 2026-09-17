"""
ArthoBodh: Bengali Word Sense Disambiguation (WSD) System
Module: Visualizations for Dataset Statistics

    python -m src.visualize_stats
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Use clean styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

from src import config
splits_file = config.SPLITS_PATH

with open(splits_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

train = data['train']
val = data['val']
test = data['test']
catalog = data['catalog']
all_records = train + val + test

vis_dir = str(config.PLOTS_DIR)
os.makedirs(vis_dir, exist_ok=True)

# 1. Distribution of Senses per Target Word (3 vs 4)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
sense_counts = [len(d['senses']) for d in catalog.values()]
c3 = sense_counts.count(3)
c4 = sense_counts.count(4)
bars = ax.bar(['3 Senses', '4 Senses'], [c3, c4], color=['#2b5c8f', '#d95f02'], width=0.5, edgecolor='black')
ax.set_ylabel('Number of Target Words', fontsize=12)
ax.set_title('Bengali WSD Dataset: Word Complexity Distribution', fontsize=13, fontweight='bold')
ax.set_ylim(0, 70)
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, y + 1.5, f'{y} words', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(vis_dir, 'word_complexity_distribution.png'))
plt.close(fig)

# 2. Train / Val / Test Split Distribution
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
split_labels = ['Train (70%)', 'Validation (10%)', 'Test (20%)']
split_counts = [len(train), len(val), len(test)]
colors = ['#1b9e77', '#d95f02', '#7570b3']
bars = ax.bar(split_labels, split_counts, color=colors, width=0.5, edgecolor='black')
ax.set_ylabel('Number of Context Paragraphs', fontsize=12)
ax.set_title('Dataset Split Distribution (Stratified by Sense)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 2800)
for bar in bars:
    y = bar.get_height()
    pct = y / len(all_records) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, y + 50, f'{y}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(vis_dir, 'dataset_split_distribution.png'))
plt.close(fig)

# 3. Context Length Distribution (Word Counts & Character Counts)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
word_lens = [r['word_len'] for r in all_records]
char_lens = [r['char_len'] for r in all_records]

ax1.hist(word_lens, bins=30, color='#386cb0', edgecolor='black', alpha=0.8)
ax1.axvline(np.mean(word_lens), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(word_lens):.1f}')
ax1.axvline(np.median(word_lens), color='green', linestyle='-', linewidth=1.5, label=f'Median: {np.median(word_lens):.1f}')
ax1.set_xlabel('Length in Words (Context Paragraph)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Context Lengths (Words)', fontsize=12, fontweight='bold')
ax1.legend(frameon=True)

ax2.hist(char_lens, bins=30, color='#f0027f', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(char_lens), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(char_lens):.1f}')
ax2.axvline(np.median(char_lens), color='green', linestyle='-', linewidth=1.5, label=f'Median: {np.median(char_lens):.1f}')
ax2.set_xlabel('Length in Characters', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Distribution of Context Lengths (Characters)', fontsize=12, fontweight='bold')
ax2.legend(frameon=True)

plt.tight_layout()
fig.savefig(os.path.join(vis_dir, 'context_length_distribution.png'))
plt.close(fig)

# 4. Examples per Target Word (Distribution across 100 words)
fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
from collections import Counter
word_counts = Counter([r['folder'] for r in all_records])
# 55 words have 30 examples, 44 words have 40 examples, 1 word has 41 examples
freq_counts = Counter(word_counts.values())
bars = ax.bar([f'{k} Examples\n({k//10} Senses)' for k in sorted(freq_counts.keys())],
              [freq_counts[k] for k in sorted(freq_counts.keys())],
              color=['#4daf4a', '#377eb8', '#984ea3'], width=0.45, edgecolor='black')
ax.set_ylabel('Number of Target Words', fontsize=12)
ax.set_title('Number of Context Examples per Target Word', fontsize=13, fontweight='bold')
for bar in bars:
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, y + 1, f'{y} words', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, 65)
plt.tight_layout()
fig.savefig(os.path.join(vis_dir, 'examples_per_target_word.png'))
plt.close(fig)

print(f"All visualizations saved successfully in {vis_dir}")
