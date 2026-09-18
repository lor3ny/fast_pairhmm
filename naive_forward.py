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

  2. PairHMM(read, hap, model="gatk").gatk_forward()
                                 -- the semi-global model used by GATK
                                 HaplotypeCaller. The read must be consumed
                                 entirely; the haplotype may be entered and
                                 exited at any position. The constructor
                                 loads per-base quality scores from
                                 gatk_params.yaml and builds the
                                 per-read-position transition matrices
                                 (they vary by read position, unlike the
                                 naive model's fixed one).

Conventions used throughout:
  i indexes the read      (rows,    1..m)
  j indexes the haplotype (columns, 1..n)
  M[i][j] = total probability of all alignments of read[:i] to hap[:j]
            that end with read[i-1] paired to hap[j-1]
  I[i][j] = ... that end with read[i-1] paired to a gap
  D[i][j] = ... that end with hap[j-1] paired to a gap
"""

from math import log10
from pathlib import Path
import yaml

DEFAULT_GLOBAL_PARAMS = Path(__file__).resolve().parent / "global_params.yaml"
DEFAULT_GATK_PARAMS = Path(__file__).resolve().parent / "gatk_params.yaml"

# GATK offsets the whole matrix by this constant so that products of
# thousands of small probabilities stay inside double-precision range.
# It is divided out again at the end, so it does not change the answer.
INITIAL_CONSTANT = 2.0 ** 1020


class PairHMM:

    def __init__(self, read, hap, model="naive", params_path=None):
        self.read = read
        self.hap = hap
        self.model = model
        m = len(read)

        if model == "gatk":
            path = params_path or DEFAULT_GATK_PARAMS
            self.transition_matrices, self.eps_sub = self._gatk_parameters(m, path)
        elif model == "naive":
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

    def _gatk_parameters(self, m, params_path):
        """Load and prepare every parameter gatk_forward() needs: a per-
        read-position transition matrix and substitution error
        probability, for a read of length m. Returns (transition_matrices,
        eps_sub), each a list of length m indexed by read position i-1."""
        with open(params_path) as f:
            config = yaml.safe_load(f)
        base_q, ins_q, del_q, gcp = (
            config["base_q"], config["ins_q"], config["del_q"], config["gcp"],
        )

        eps_sub = [self._phred_to_prob(base_q)] * m
        t_mi = [self._phred_to_prob(ins_q)] * m
        t_md = [self._phred_to_prob(del_q)] * m
        t_mm = [1.0 - (t_mi[k] + t_md[k]) for k in range(m)]
        t_ii = t_dd = self._phred_to_prob(gcp)
        t_im = t_dm = 1.0 - t_ii

        transition_matrices = [
            {
                "M": {"M": t_mm[k], "I": t_mi[k], "D": t_md[k]},
                "I": {"M": t_im, "I": t_ii},
                "D": {"M": t_dm, "D": t_dd},
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
    #? insertion/deletion quality scores (uniform across the read, loaded
    #? from a YAML file). Returns (log10 P(read|hap), M, I, D)."""
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
            for j in range(1, n + 1):
                e_m = (1.0 - e) if read[i - 1] == hap[j - 1] else e / 3.0
                M[i][j] = e_m * (
                    T["M"]["M"] * M[i - 1][j - 1]
                    + T["I"]["M"] * I[i - 1][j - 1]
                    + T["D"]["M"] * D[i - 1][j - 1]
                )
                I[i][j] = T["M"]["I"] * M[i - 1][j] + T["I"]["I"] * I[i - 1][j]
                D[i][j] = T["M"]["D"] * M[i][j - 1] + T["D"]["D"] * D[i][j - 1]

        #! I don't understand why it returns the SUM.
        # free exit: sum across the whole last row
        total = sum(M[m][j] + I[m][j] for j in range(1, n + 1))
        return log10(total) - log10(INITIAL_CONSTANT), M, I, D



def show(name, mat, read, hap, fmt="10.6f"):
    width = int(fmt.split(".")[0])
    print(f"  {name}")
    print("        " + "".join(f"{c:>{width}}" for c in "-" + hap))
    for i, row in enumerate(mat):
        label = "-" if i == 0 else read[i - 1]
        print(f"     {label}  " + "".join(f"{v:{fmt}}" for v in row))


if __name__ == "__main__":

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

