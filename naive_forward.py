"""
Pair HMM forward algorithm, from scratch.

Build a PairHMM(read, hap, model=...) -- the constructor loads that
model's parameters right away -- then call its forward algorithm (states
M, I, D):

  1. PairHMM(read, hap, model="naive").naive_forward()
                                 -- the textbook global model (Durbin et
                                 al., ch. 4). Both sequences must be
                                 consumed entirely. The constructor loads
                                 the transition matrix and emission
                                 probabilities from global_params.yaml.

  2. PairHMM(read, hap, model="gatk", quals=None).gatk_forward()
                                 -- the semi-global model used by GATK
                                 HaplotypeCaller. The read must be consumed
                                 entirely; the haplotype may be entered and
                                 exited at any position. The constructor
                                 takes the quality scores either from
                                 `quals` (one Phred score per read base, as
                                 GATK does) or, if quals is None, from
                                 gatk_params.yaml (one value repeated along
                                 the read), and builds the per-read-position
                                 transition matrices (they vary by read
                                 position, unlike the naive model's fixed
                                 one).

Command line:

  python3 naive_forward.py
      prints a small demo of both models.

  python3 naive_forward.py --dump pairhmm_dump.txt --out ours.tsv
      runs gatk_forward() on every read/haplotype pair of a dump written
      by `gatk HaplotypeCaller --pair-hmm-results-file` and writes one
      log10 P(read|hap) per pair to ours.tsv. compare_with_gatk.py then
      compares that file with GATK's own values.

Conventions used throughout:
  i indexes the read      (rows,    1..m)
  j indexes the haplotype (columns, 1..n)
  M[i][j] = total probability of all alignments of read[:i] to hap[:j]
            that end with read[i-1] paired to hap[j-1]
  I[i][j] = ... that end with read[i-1] paired to a gap
  D[i][j] = ... that end with hap[j-1] paired to a gap
"""

import argparse
from math import log10
from pathlib import Path
import yaml

DEFAULT_GLOBAL_PARAMS = Path(__file__).resolve().parent / "global_params.yaml"
DEFAULT_GATK_PARAMS = Path(__file__).resolve().parent / "gatk_params.yaml"

# GATK offsets the whole matrix by this constant so that products of
# thousands of small probabilities stay inside double-precision range.
# It is divided out again at the end, so it does not change the answer.
INITIAL_CONSTANT = 2.0 ** 1020

QUAL_KEYS = ("base", "ins", "del", "gcp")


class PairHMM:

    def __init__(self, read, hap, model="naive", params_path=None, quals=None):
        """quals (model="gatk" only): optional dict of per-read-position
        Phred scores with keys "base", "ins", "del" and "gcp", each a list
        as long as the read. When given, gatk_params.yaml is not read."""
        self.read = read
        self.hap = hap
        self.model = model
        m = len(read)

        if model == "gatk":
            path = params_path or DEFAULT_GATK_PARAMS
            self.transition_matrices, self.eps_sub = self._gatk_parameters(m, path, quals)
        elif model == "naive":
            if quals is not None:
                raise ValueError("per-base quals are only used by model='gatk'")
            path = params_path or DEFAULT_GLOBAL_PARAMS
            (self.transition_matrix, self.p_match,
             self.p_mismatch, self.gap_emit) = self._naive_parameters(path)
        else:
            raise ValueError(f"unknown model: {model!r} (expected 'naive' or 'gatk')")

        
#* --------------------------------------------------------------------------------
#* PARAMETER LOADING AND PREPARATION
#* --------------------------------------------------------------------------------

    @staticmethod
    def _phred_to_prob(q):
        """Phred score -> error probability."""
        return 10.0 ** (-q / 10.0)

    @staticmethod
    def _naive_parameters(params_path):
        """Load and prepare every parameter naive_forward() needs: the
        transition matrix plus match/mismatch/gap emission probabilities.
        Returns (transition_matrix, p_match, p_mismatch, gap_emit)."""
        with open(params_path) as f:
            config = yaml.safe_load(f)
        transition_matrix = config["transition"]
        p_match = config["emission"]["match"]
        gap_emit = config["emission"]["gap"]
        p_mismatch = (1.0 - p_match) / 3.0
        return transition_matrix, p_match, p_mismatch, gap_emit

    def _gatk_parameters(self, m, params_path, quals=None):
        """Load and prepare every parameter gatk_forward() needs: a per-
        read-position transition matrix and substitution error
        probability, for a read of length m. Returns (transition_matrices,
        eps_sub), each a list of length m indexed by read position i-1.

        The Phred scores come from `quals` if given (per read position,
        e.g. from a GATK PairHMM dump); otherwise the YAML file's single
        values are repeated along the read."""
        if quals is None:
            with open(params_path) as f:
                config = yaml.safe_load(f)
            quals = {
                "base": [config["base_q"]] * m,
                "ins": [config["ins_q"]] * m,
                "del": [config["del_q"]] * m,
                "gcp": [config["gcp"]] * m,
            }
        for key in QUAL_KEYS:
            if len(quals[key]) != m:
                raise ValueError(
                    f"quals[{key!r}] has {len(quals[key])} values, but the read has {m} bases"
                )

        eps_sub = [self._phred_to_prob(q) for q in quals["base"]]
        t_mi = [self._phred_to_prob(q) for q in quals["ins"]]
        t_md = [self._phred_to_prob(q) for q in quals["del"]]
        t_mm = [1.0 - (t_mi[k] + t_md[k]) for k in range(m)]
        # The gap continuation penalty can also differ per read position.
        t_ii = t_dd = [self._phred_to_prob(q) for q in quals["gcp"]]
        t_im = t_dm = [1.0 - t_ii[k] for k in range(m)]

        transition_matrices = [
            {
                "M": {"M": t_mm[k], "I": t_mi[k], "D": t_md[k]},
                "I": {"M": t_im[k], "I": t_ii[k]},
                "D": {"M": t_dm[k], "D": t_dd[k]},
            }
            for k in range(m)
        ]
        return transition_matrices, eps_sub


#* --------------------------------------------------------------------------------
#* FORWARD ALGORITHMS
#* --------------------------------------------------------------------------------

    #? ----------------------------------------------------------------------------
    #? Textbook global forward algorithm (Durbin et al., ch. 4): sum
    #? the probability of every alignment that consumes both sequences
    #? entirely. Returns (total, M, I, D)."""
    #? ----------------------------------------------------------------------------
    def naive_forward(self):
        """Textbook global forward algorithm (Durbin et al., ch. 4): sum
        the probability of every alignment that consumes both sequences
        entirely. Returns (total, M, I, D)."""
        if self.model != "naive":
            raise ValueError("naive_forward() requires PairHMM(..., model='naive')")

        #! PARAMETERS
        read, hap = self.read, self.hap
        m, n = len(read), len(hap)
        T = self.transition_matrix
        p_match, p_mismatch, gap_emit = self.p_match, self.p_mismatch, self.gap_emit

        #! DP MATRICES INIT
        M = [[0.0] * (n + 1) for _ in range(m + 1)]
        I = [[0.0] * (n + 1) for _ in range(m + 1)]
        D = [[0.0] * (n + 1) for _ in range(m + 1)]
        M[0][0] = 1.0  # Begin is identified with the match state


        #! DP INDUCTION
        for i in range(m + 1):
            for j in range(n + 1):

                #! Skip the first step, already initialized above to be MATCH
                if i == 0 and j == 0:
                    continue

                #! Induction step, we compute the probability of a MATCH, given Xi for every Yj
                if i > 0 and j > 0:
                    e_m = p_match if read[i - 1] == hap[j - 1] else p_mismatch

                    #* Mij = match probability * (sum of all ways to get to Mij from previous states)
                    #* transition from M to M, I to M, D to M. Considering the previous states probability (i-1, j-1).
                    M[i][j] = e_m * (
                        T["M"]["M"] * M[i - 1][j - 1]
                        + T["I"]["M"] * I[i - 1][j - 1]
                        + T["D"]["M"] * D[i - 1][j - 1]
                    )

                #! Induction step, we compute the probability of an INSERTION, given Xi for every Yj
                if i > 0:
                    I[i][j] = gap_emit * (
                        T["M"]["I"] * M[i - 1][j] + T["I"]["I"] * I[i - 1][j]
                    )

                #! Induction step, we compute the probability of a DELETION, given Xi for every Yj
                if j > 0:
                    D[i][j] = gap_emit * (
                        T["M"]["D"] * M[i][j - 1] + T["D"]["D"] * D[i][j - 1]
                    )


        total = M[m][n] + I[m][n] + D[m][n]
        return total, M, I, D


    #? ----------------------------------------------------------------------------
    #? Semi-global GATK HaplotypeCaller forward algorithm: the read is
    #? consumed entirely, the haplotype may be entered/exited anywhere,
    #? and the transition matrix varies per read position from per-base
    #? insertion/deletion quality scores (from `quals`, or uniform across
    #? the read from a YAML file). Returns (log10 P(read|hap), M, I, D)."""
    #? ----------------------------------------------------------------------------
    def gatk_forward(self):

        if self.model != "gatk":
            raise ValueError("gatk_forward() requires PairHMM(..., model='gatk')")

        #! PARAMETERS
        read, hap = self.read, self.hap
        m, n = len(read), len(hap)
        transition_matrices, eps_sub = self.transition_matrices, self.eps_sub

        #! DP MATRICES INIT
        M = [[0.0] * (n + 1) for _ in range(m + 1)]
        I = [[0.0] * (n + 1) for _ in range(m + 1)]
        D = [[0.0] * (n + 1) for _ in range(m + 1)]
        for j in range(n + 1):  # free entry into every haplotype position
            D[0][j] = INITIAL_CONSTANT / n


        #! DP INDUCTION
        for i in range(1, m + 1):
            T = transition_matrices[i - 1]
            e = eps_sub[i - 1]
            x = read[i - 1]
            for j in range(1, n + 1):
                y = hap[j - 1]
                # Like GATK, an N in either sequence counts as a match.
                e_m = (1.0 - e) if (x == y or x == "N" or y == "N") else e / 3.0
                M[i][j] = e_m * (
                    T["M"]["M"] * M[i - 1][j - 1]
                    + T["I"]["M"] * I[i - 1][j - 1]
                    + T["D"]["M"] * D[i - 1][j - 1]
                )
                I[i][j] = T["M"]["I"] * M[i - 1][j] + T["I"]["I"] * I[i - 1][j]
                D[i][j] = T["M"]["D"] * M[i][j - 1] + T["D"]["D"] * D[i][j - 1]

        #! Free exit: the read may end at ANY haplotype position j. Alignments
        #! that end at different j are mutually exclusive, so their
        #! probabilities add up. D is left out because a read cannot end on a
        #! deletion: its last base must be emitted by M or I. GATK's
        #! LoglessPairHMM computes exactly this sum.
        total = sum(M[m][j] + I[m][j] for j in range(1, n + 1))
        return log10(total) - log10(INITIAL_CONSTANT), M, I, D


#* --------------------------------------------------------------------------------
#* BATCH RUN ON A GATK PAIRHMM DUMP
#* --------------------------------------------------------------------------------

def _fastq_to_phred(qual_string):
    """FASTQ quality string (Phred+33) -> list of Phred scores."""
    return [ord(c) - 33 for c in qual_string]


def read_pairhmm_inputs(path):
    """Parse the inputs of the file written by
    `gatk HaplotypeCaller --pair-hmm-results-file`.

    GATK writes one header line,
        # hap-bases read-bases read-qual read-ins-qual read-del-qual gcp expected-result
    then one line per read/haplotype pair with those 7 space-separated
    fields: the haplotype, the read (already trimmed by GATK) and four
    FASTQ-encoded per-base quality strings. GATK's own result (last
    field) is ignored here; compare_with_gatk.py reads it.

    Yields (line_number, read, hap, quals)."""
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"{path}:{line_no}: expected 7 fields, found {len(fields)}")
            hap, read, base_q, ins_q, del_q, gcp, _ = fields
            quals = dict(zip(QUAL_KEYS, map(_fastq_to_phred, (base_q, ins_q, del_q, gcp))))
            yield line_no, read, hap, quals


def run_on_dump(dump_path, out_path, limit=None):
    """Run gatk_forward() on every read/haplotype pair of a GATK PairHMM
    dump and write one row per pair to out_path (TSV with columns
    dump_line, read_len, hap_len, log10_likelihood). The likelihood is
    written with full double precision (repr), so no digits are lost
    before the comparison."""
    n_pairs = 0
    with open(out_path, "w") as f:
        f.write("dump_line\tread_len\thap_len\tlog10_likelihood\n")
        for k, (line_no, read, hap, quals) in enumerate(read_pairhmm_inputs(dump_path)):
            if limit is not None and k >= limit:
                break
            ll, _, _, _ = PairHMM(read, hap, model="gatk", quals=quals).gatk_forward()
            f.write(f"{line_no}\t{len(read)}\t{len(hap)}\t{ll!r}\n")
            n_pairs += 1
    print(f"gatk_forward(): {n_pairs} read/haplotype pairs from {dump_path} -> {out_path}")


def show(name, mat, read, hap, fmt="10.6f"):
    width = int(fmt.split(".")[0])
    print(f"  {name}")
    print("        " + "".join(f"{c:>{width}}" for c in "-" + hap))
    for i, row in enumerate(mat):
        label = "-" if i == 0 else read[i - 1]
        print(f"     {label}  " + "".join(f"{v:{fmt}}" for v in row))


def demo():
    read, hap = "ACT", "AACTGCT"

    print(f"Global forward:  read {read}  vs  haplotype {hap}\n")
    naive_hmm = PairHMM(read, hap, model="naive")
    total, M, I, D = naive_hmm.naive_forward()
    show("M (read base on haplotype base)", M, read, hap)
    show("I (read base on gap)", I, read, hap)
    show("D (haplotype base on gap)", D, read, hap)
    print(f"\n  forward total        = {total:.6f}")


    print(f"\nGATK forward:  read {read}  vs  haplotype {hap}\n")
    gatk_hmm = PairHMM(read, hap, model="gatk")
    ll, M, I, D = gatk_hmm.gatk_forward()
    show("M (read base on haplotype base)", M, read, hap, fmt="14.3e")
    show("I (read base on gap)", I, read, hap, fmt="14.3e")
    show("D (haplotype base on gap)", D, read, hap, fmt="14.3e")
    print(f"\n  log10 P(read|hap)    = {ll:.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pair HMM forward algorithm. Without options, runs the small "
                    "demo (needs global_params.yaml and gatk_params.yaml).")
    parser.add_argument("--dump", metavar="DUMP",
                        help="file written by gatk HaplotypeCaller --pair-hmm-results-file; "
                             "run gatk_forward() on each read/haplotype pair")
    parser.add_argument("--out", metavar="TSV",
                        help="with --dump: write one log10 likelihood per pair here")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="with --dump: only compute the first N pairs")
    args = parser.parse_args()

    if args.dump:
        if not args.out:
            parser.error("--dump requires --out")
        run_on_dump(args.dump, args.out, args.limit)
    else:
        demo()
