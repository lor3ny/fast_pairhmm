#!/usr/bin/env python3
"""
Build a tiny synthetic dataset for testing GATK HaplotypeCaller.

Creates a random 20 kb reference, plants known SNPs and small indels
(both het and hom) into a diploid "sample", and simulates 30x paired-end
150 bp Illumina-like reads from it.

Outputs: ref.fa, reads_R1.fq, reads_R2.fq, truth.tsv
"""
import random

random.seed(7)
REF_LEN = 20_000
READ_LEN = 150
FRAG_MEAN, FRAG_SD = 400, 40
COVERAGE = 30            # total depth across both haplotypes
ERROR_RATE = 0.001       # per-base sequencing error rate
BASES = "ACGT"
COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s):
    return s.translate(COMP)[::-1]


def left_align_insertion(ref, p, ins):
    """Insertion of `ins` right after ref[p] (0-based), written in its
    leftmost equivalent form, as GATK and `bcftools norm` report it: while
    the anchor base equals the last inserted base, shifting the insertion
    one base left gives the same haplotype. Returns (POS, REF, ALT)."""
    while p > 0 and ref[p] == ins[-1]:
        ins = ref[p] + ins[:-1]
        p -= 1
    return p + 1, ref[p], ref[p] + ins


def left_align_deletion(ref, p, k):
    """Deletion of the k bases after ref[p] (0-based), in its leftmost
    equivalent form. Returns (POS, REF, ALT)."""
    while p > 0 and ref[p] == ref[p + k]:
        p -= 1
    return p + 1, ref[p:p + k + 1], ref[p]


def add_errors(seq):
    return "".join(
        random.choice([x for x in BASES if x != b]) if random.random() < ERROR_RATE else b
        for b in seq
    )


# --- reference -------------------------------------------------------------
ref = "".join(random.choice(BASES) for _ in range(REF_LEN))
with open("ref.fa", "w") as f:
    f.write(">chr1\n")
    for i in range(0, REF_LEN, 60):
        f.write(ref[i:i + 60] + "\n")

# --- plant variants into two haplotypes ------------------------------------
hap = [list(ref), list(ref)]   # list of strings -> indels are easy
truth = []
for n, p in enumerate(range(600, REF_LEN - 600, 600)):   # 0-based positions
    zyg = "HOM" if n % 3 == 0 else "HET"
    kind = "INS" if n % 7 == 3 else "DEL" if n % 7 == 5 else "SNP"
    which = [0, 1] if zyg == "HOM" else [0]
    if kind == "SNP":
        alt = random.choice([b for b in BASES if b != ref[p]])
        for h in which:
            hap[h][p] = alt
        truth.append((p + 1, ref[p], alt, zyg))
    elif kind == "INS":
        ins = "".join(random.choice(BASES) for _ in range(3))
        for h in which:
            hap[h][p] = ref[p] + ins
        truth.append(left_align_insertion(ref, p, ins) + (zyg,))
    else:  # 2 bp deletion after anchor base
        for h in which:
            hap[h][p + 1] = ""
            hap[h][p + 2] = ""
        truth.append(left_align_deletion(ref, p, 2) + (zyg,))

with open("truth.tsv", "w") as f:
    for pos, r, a, z in truth:
        f.write(f"{pos}\t{r}\t{a}\t{z}\n")

# --- simulate paired-end reads ---------------------------------------------
qual = "?" * READ_LEN  # Phred 30
rid = 0
with open("reads_R1.fq", "w") as r1, open("reads_R2.fq", "w") as r2:
    for h in hap:
        seq = "".join(h)
        n_pairs = int((COVERAGE / 2) * len(seq) / (2 * READ_LEN))
        for _ in range(n_pairs):
            frag_len = max(READ_LEN, int(random.gauss(FRAG_MEAN, FRAG_SD)))
            start = random.randint(0, len(seq) - frag_len)
            frag = seq[start:start + frag_len]
            if random.random() < 0.5:
                frag = revcomp(frag)
            rid += 1
            r1.write(f"@read{rid}\n{add_errors(frag[:READ_LEN])}\n+\n{qual}\n")
            r2.write(f"@read{rid}\n{add_errors(revcomp(frag)[:READ_LEN])}\n+\n{qual}\n")

print(f"Wrote ref.fa ({REF_LEN} bp), {rid} read pairs, {len(truth)} truth variants")
