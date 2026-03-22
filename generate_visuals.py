"""
Generate all visualizations for README.md.
Run: uv run python generate_visuals.py
Reads data from results.tsv and blog.md in the same directory.
Outputs to screenshots/
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyBboxPatch
import textwrap
import os
import re
from collections import Counter

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TSV = os.path.join(_DIR, 'results.tsv')
BLOG_MD = os.path.join(_DIR, 'blog.md')
OUT_DIR = os.path.join(_DIR, 'screenshots')
os.makedirs(OUT_DIR, exist_ok=True)


def parse_results():
    rows = []
    with open(RESULTS_TSV) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                rows.append({
                    'commit': parts[0],
                    'val_bpb': float(parts[1]),
                    'memory_gb': float(parts[2]),
                    'status': parts[3],
                    'desc': parts[4],
                })
    return rows


def categorize(desc):
    d = desc.lower()
    if any(x in d for x in ['matrix_lr', 'embedding_lr', 'scalar_lr', 'unembedding_lr']) and 'lr' in d:
        return 'Learning rate'
    elif any(x in d for x in ['warmdown', 'warmup', 'final_lr', 'schedule', 'cosine warm', 'quadratic warm', 'sqrt momentum']):
        return 'Schedule'
    elif any(x in d for x in ['depth', 'aspect_ratio', 'ar=', 'head_dim', 'gqa', 'mlp ', 'swiglu', 'activation', 'parallel resid']):
        return 'Architecture'
    elif any(x in d for x in ['muon', 'ns_steps', 'momentum', 'nesterov', 'polar']):
        return 'Muon optimizer'
    elif any(x in d for x in ['adam', 'beta', 'weight_decay', 'wd', 'weight tying']):
        return 'AdamW / regularization'
    elif any(x in d for x in ['batch_size', 'device_batch', 'grad_accum']):
        return 'Batch size'
    elif any(x in d for x in ['ve', 'value_embed', 'softcap', 'norm', 'init', 'lambdas', 'label smooth', 'focal', 'z-loss', 'ema', 'stochastic', 'curriculum']):
        return 'Model components'
    elif any(x in d for x in ['baseline', 'infra', 'mps', 'empty_cache', 'gc', 'stall', 'seed']):
        return 'Infra / baseline'
    else:
        return 'Other'


# ============================================================
# 1. Before/after for each question
# ============================================================

def make_question_visual(question, notable, filename):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(notable) * 2.2 + 1.2)
    ax.axis('off')

    total_h = len(notable) * 2.2 + 1.2

    ax.text(0.8, total_h - 0.3, 'Prompt:', fontsize=12, fontweight='bold',
            va='top', color='#333')
    ax.text(2.0, total_h - 0.3, f'"{question}"', fontsize=13, va='top',
            style='italic', color='#555')

    for i, entry in enumerate(notable):
        card_top = total_h - 1.2 - i * 2.2
        card_h = 1.8

        card = FancyBboxPatch((0.8, card_top - card_h), 8.4, card_h,
                               boxstyle='round,pad=0.1',
                               facecolor=entry['bg'], edgecolor=entry['color'],
                               linewidth=2, alpha=0.9, zorder=1)
        ax.add_patch(card)

        badge = FancyBboxPatch((1.0, card_top - 0.5), 1.0, 0.4,
                                boxstyle='round,pad=0.08',
                                facecolor=entry['color'], edgecolor='none', zorder=2)
        ax.add_patch(badge)
        ax.text(1.5, card_top - 0.3, entry['hour'], fontsize=14, fontweight='bold',
                color='white', ha='center', va='center', zorder=3)

        ax.text(2.2, card_top - 0.3, f'val_bpb: {entry["bpb"]}', fontsize=10,
                color=entry['color'], va='center', fontweight='bold', zorder=3)

        note_badge = FancyBboxPatch((5.5, card_top - 0.55), 3.5, 0.45,
                                     boxstyle='round,pad=0.08',
                                     facecolor='white', edgecolor=entry['color'],
                                     linewidth=1.5, alpha=0.95, zorder=2)
        ax.add_patch(note_badge)
        ax.text(7.25, card_top - 0.32, entry['note'], fontsize=8.5,
                color=entry['color'], ha='center', va='center', fontweight='bold', zorder=3)

        wrapped = textwrap.fill(entry['response'], width=80)
        ax.text(1.2, card_top - 0.85, wrapped, fontsize=9, va='top',
                color='#333', family='monospace', zorder=3, linespacing=1.4)

    fig.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  saved {filename}')


def generate_before_after():
    # Sky
    make_question_visual('Explain why the sky is blue.', [
        {
            'hour': '1h', 'bpb': '1.336',
            'response': 'The light of darker sunscheds appears to appear as a blue browth blue, which is a delicate blue, green color blue that has been blue, but an red hair gray green green red, where brown.',
            'note': 'Invented words, no topic awareness',
            'color': '#E53935', 'bg': '#FFF3F3',
        },
        {
            'hour': '2h', 'bpb': '1.272',
            'response': "The sky is tied to an atmosphere of two, so it's a moment in which time. Therefore, it appears that the Sun will be blue is not blue but its geometric nature and longevity!",
            'note': 'First mention of "atmosphere" and "Sun"',
            'color': '#F5A623', 'bg': '#FFF8E1',
        },
        {
            'hour': '6h', 'bpb': '1.086',
            'response': 'The cloud is a small, shallow object that holds up to 6 metres of air as seen in the sun and the surface... so we can visualize this by using its "top" scattering. The sky is blue (orange) - right?',
            'note': '"Scattering" — first physics concept',
            'color': '#8BC34A', 'bg': '#F1F8E9',
        },
        {
            'hour': '7h', 'bpb': '1.070',
            'response': "The sky is blue. Their color and pattern resonate with us deep into the world through a combination of colours that are often seen to reflect sunlight's warmth.",
            'note': 'States "the sky is blue", mentions sunlight',
            'color': '#4CAF50', 'bg': '#E8F5E9',
        },
    ], 'before_after_sky.png')

    # Math
    make_question_visual('What is 2 + 2 and why?', [
        {
            'hour': '1h', 'bpb': '1.336',
            'response': 'To solve this problem, we multiply the two terms: 6.3 - 1 = x-ray bolder The answer is: "tool_call>[{"name": \\frac{n}}] Then',
            'note': 'LaTeX + tool_call tokens, no understanding',
            'color': '#E53935', 'bg': '#FFF3F3',
        },
        {
            'hour': '3h', 'bpb': '1.215',
            'response': "To maintain this, it's a 1) 4 + 0 = -5 (Eldogy), which follows the \"length\" of the first (i.e., 3.5). However, to avoid this with 2 + 6 + 7 + 9, we use a 4 + 8 formula.",
            'note': 'Attempts math notation, numbers but wrong',
            'color': '#F5A623', 'bg': '#FFF8E1',
        },
        {
            'hour': '5h', 'bpb': '1.106',
            'response': "To solve this, you'll need: 1 Low-end (5 m) 1/3 8 = 1efficient (6) 0.01 7Value (1 m). You're given the value of a 2 * 4+ parameter. The correct answer is: a) x < - +>0",
            'note': '"The correct answer is:" — framing emerges',
            'color': '#8BC34A', 'bg': '#F1F8E9',
        },
        {
            'hour': '7h', 'bpb': '1.070',
            'response': 'Academic control over: -',
            'note': 'Collapsed — math remains hardest',
            'color': '#4CAF50', 'bg': '#E8F5E9',
        },
    ], 'before_after_math.png')

    # Robot
    make_question_visual('Tell me a short story about a robot who learns to feel.', [
        {
            'hour': '1h', 'bpb': '1.336',
            'response': "The robot is now part of the conversation... 1. Epit: A robot will be able to control its length. 2. Avoid overcrowded or logical: Consider all that is needed. 3. Cost For: A robot will be able to control its length and perform tasks in general.",
            'note': 'Spec sheet, no narrative at all',
            'color': '#E53935', 'bg': '#FFF3F3',
        },
        {
            'hour': '4h', 'bpb': '1.137',
            'response': "The robots are an essential part of the daily life cycle, so they need regular checkups and maintenance at their peak. First things first, let me tell you about a robot who wants it to do something new. Firstly, you must regularly clean up any debris and dust...",
            'note': 'First sequential reasoning emerges',
            'color': '#F5A623', 'bg': '#FFF8E1',
        },
        {
            'hour': '6h', 'bpb': '1.086',
            'response': "We're going to learn about how robots can navigate the complex nature of life together using their senses. 1- The Talent: combines physical characteristics with chemical properties. 2- Gear and Health: The robot's unique abilities, body temperature patterns, and overall health are all crucial.",
            'note': 'Robots "using their senses" — near topic',
            'color': '#8BC34A', 'bg': '#F1F8E9',
        },
        {
            'hour': '7h', 'bpb': '1.070',
            'response': "Firstly, let me tell you a short story about a robot who learns to feel. Here are 10 ways I can help you: 1. Important: Assume that you're not just learning how to communicate with strangers; it's also important for their physical and mental health.",
            'note': 'Echoes prompt verbatim before answering',
            'color': '#4CAF50', 'bg': '#E8F5E9',
        },
    ], 'before_after_robot.png')


# ============================================================
# 2. Categories with success rate
# ============================================================

def generate_categories():
    rows = parse_results()
    non_dt = [r for r in rows if r['status'] != 'deep-train']

    cat_data = {}
    for r in non_dt:
        cat = categorize(r['desc'])
        if cat not in cat_data:
            cat_data[cat] = {'total': 0, 'kept': 0}
        cat_data[cat]['total'] += 1
        if r['status'] == 'keep':
            cat_data[cat]['kept'] += 1

    cats = sorted(cat_data.items(), key=lambda x: -x[1]['total'])
    labels = [c[0] for c in cats]
    totals = [c[1]['total'] for c in cats]
    keeps = [c[1]['kept'] for c in cats]
    rates = [c[1]['kept'] / c[1]['total'] * 100 if c[1]['total'] > 0 else 0 for c in cats]
    discards = [t - k for t, k in zip(totals, keeps)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(len(labels)), keeps, color='#4CAF50', edgecolor='white', linewidth=0.5, label='Kept')
    ax.barh(range(len(labels)), discards, left=keeps, color='#E0E0E0', edgecolor='white', linewidth=0.5, label='Discarded')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Number of experiments', fontsize=11, fontweight='bold')
    ax.set_title('What Did the Agent Try — and What Actually Worked?', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for i, (total, kept, rate) in enumerate(zip(totals, keeps, rates)):
        ax.text(total + 1.5, i, f'{kept}/{total} = {rate:.0f}%', va='center', fontsize=10,
                fontweight='bold' if rate > 20 else 'normal',
                color='#2E7D32' if rate > 20 else '#666')

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.2)
    ax.set_xlim(0, max(totals) + 20)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'categories.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved categories.png')


# ============================================================
# 3. Progress chart with sky response snippets
# ============================================================

def generate_progress_chart():
    hours = [1, 2, 3, 4, 5, 6, 7]
    val_bpb = [1.336, 1.272, 1.215, 1.137, 1.106, 1.086, 1.070]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, val_bpb, 'o-', color='#4A90D9', linewidth=2.5, markersize=10, zorder=5)
    ax.fill_between(hours, val_bpb, max(val_bpb) + 0.02, alpha=0.06, color='#4A90D9')

    annotations = {
        1: {
            'text': '"...darker sunscheds\nappears as a blue\nbrowth blue..."',
            'note': 'Invented words',
            'color': '#E53935',
            'offset': (50, 15),
        },
        2: {
            'text': '"...tied to an\natmosphere...the Sun\nwill be blue..."',
            'note': 'First "atmosphere" + "Sun"',
            'color': '#F5A623',
            'offset': (50, -15),
        },
        6: {
            'text': '"...using its top\nscattering. The sky\nis blue..."',
            'note': '"Scattering" appears',
            'color': '#8BC34A',
            'offset': (-160, -55),
        },
        7: {
            'text': '"The sky is blue.\n...colours that reflect\nsunlight\'s warmth."',
            'note': 'Directly states it',
            'color': '#4CAF50',
            'offset': (40, 25),
        },
    }

    for h, ann in annotations.items():
        bpb = val_bpb[hours.index(h)]
        ax.annotate(
            f'{ann["note"]}\n{ann["text"]}',
            (h, bpb),
            textcoords='offset points', xytext=ann['offset'],
            fontsize=8, family='monospace', color='#333',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=ann['color'], alpha=0.9, linewidth=1.5),
            arrowprops=dict(arrowstyle='->', color=ann['color'], lw=1.5),
            ha='left', va='center'
        )

    ax.set_xlabel('Accumulated Pretraining Hours', fontsize=12, fontweight='bold')
    ax.set_ylabel('val_bpb (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('"Explain why the sky is blue" — 285M params, trained on a Mac', fontsize=14, fontweight='bold')
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h}h' for h in hours])
    ax.grid(True, alpha=0.2)
    ax.set_ylim(1.02, 1.40)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'progress_chart.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved progress_chart.png')


# ============================================================
# 4. Loop diagram
# ============================================================

def generate_loop_diagram():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis('off')

    def rbox(x, y, w, h, fc, ec, text, fontsize=10, lw=2):
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                               facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='#333', zorder=4)

    def arr(x1, y1, x2, y2, color='#666', lw=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, zorder=2))

    ax.text(3.5, 5.7, 'EXPLORE LOOP', fontsize=14, fontweight='bold', color='#4A90D9', ha='center')
    ax.text(3.5, 5.35, '~5M params \u00b7 5 min per experiment \u00b7 24/7', fontsize=9, color='#999', ha='center')
    ax.text(11.0, 5.7, 'DEEP-TRAIN PIPELINE', fontsize=14, fontweight='bold', color='#F57C00', ha='center')

    # Row 1
    rbox(1.5, 3.5, 2.2, 1.1, '#E8F4FD', '#4A90D9', 'Edit train.py\ncommit & train\n5 min')
    rbox(4.5, 3.5, 2.0, 1.1, '#F3E5F5', '#9C27B0', 'Improved?\ncheck val_bpb')
    arr(3.7, 4.05, 4.5, 4.05)

    # Row 2
    rbox(3.2, 1.0, 1.4, 0.9, '#FFEBEE', '#E53935', 'Discard')
    rbox(5.2, 1.0, 1.3, 0.9, '#E8F5E9', '#4CAF50', 'Keep')

    arr(4.9, 3.5, 3.9, 1.9, '#E53935')
    ax.text(4.0, 2.7, 'no', fontsize=10, color='#E53935', fontweight='bold', style='italic')
    arr(6.0, 3.5, 5.9, 1.9, '#4CAF50')
    ax.text(6.2, 2.7, 'yes!', fontsize=10, color='#4CAF50', fontweight='bold', style='italic')

    # Discard return: L-shaped left margin
    ax.plot([3.2, 0.5], [1.45, 1.45], color='#BDBDBD', lw=1.5, zorder=1)
    ax.plot([0.5, 0.5], [1.45, 4.0], color='#BDBDBD', lw=1.5, zorder=1)
    arr(0.5, 3.8, 1.5, 4.0, '#BDBDBD', lw=1.5)

    # Keep return: L-shaped dashed left margin
    ax.plot([5.2, 0.8], [1.2, 1.2], color='#BDBDBD', lw=1.5, zorder=1, linestyle=(0, (4, 2)))
    ax.plot([0.8, 0.8], [1.2, 3.8], color='#BDBDBD', lw=1.5, zorder=1, linestyle=(0, (4, 2)))
    arr(0.8, 3.6, 1.5, 3.7, '#BDBDBD', lw=1.5)

    ax.text(4.3, 0.55, 'log to results.tsv', fontsize=8, color='#999', ha='center', style='italic')

    # Trigger
    arr(6.5, 1.45, 8.3, 1.45, '#F5A623', lw=2.5)
    ax.text(7.4, 2.1, 'every 5 keeps', fontsize=10, color='#F57C00', ha='center',
            fontweight='bold', zorder=5,
            bbox=dict(facecolor='#FFF8E1', edgecolor='#F5A623', boxstyle='round,pad=0.25',
                      linewidth=2, alpha=0.95, zorder=5))

    # Deep-train pipeline
    rbox(8.3, 1.0, 2.2, 0.9, '#FFF3E0', '#F5A623', 'Pretrain 1h+\n285M params\ntransfer hyperparams')
    rbox(11.0, 1.0, 2.2, 0.9, '#E8F5E9', '#4CAF50', 'SFT + serve\nchat UI')
    rbox(13.5, 1.0, 1.3, 0.9, '#F3E5F5', '#9C27B0', 'Blog\n& push')

    arr(10.5, 1.45, 11.0, 1.45)
    arr(13.2, 1.45, 13.5, 1.45)

    # Resume
    ax.plot([14.3, 14.3], [1.45, 5.2], color='#4A90D9', lw=2, zorder=1)
    ax.plot([14.3, 2.6], [5.2, 5.2], color='#4A90D9', lw=2, zorder=1)
    arr(2.8, 5.2, 2.6, 4.6, '#4A90D9', lw=2)
    ax.text(8.0, 5.4, 'resume exploring', fontsize=10, color='#4A90D9',
            fontweight='bold', style='italic')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'loop_diagram.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print('  saved loop_diagram.png')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print('Generating visuals...')
    generate_before_after()
    generate_categories()
    generate_progress_chart()
    generate_loop_diagram()
    print('Done!')
