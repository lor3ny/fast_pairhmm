"""
Textbook global Pair HMM forward algorithm (Durbin et al., ch. 4), kept
as a reference baseline. Both sequences must be consumed entirely.

  PairHMM(read, hap).naive_forward()

The constructor loads the transition matrix and emission probabilities
from global_params.yaml (next to this script).

  python3 naive/naive_forward.py      prints the DP matrices of a small example.

Conventions used throughout:
  i indexes the read      (rows,    1..m)
  j indexes the haplotype (columns, 1..n)
  M[i][j] = total probability of all alignments of read[:i] to hap[:j]
            that end with read[i-1] paired to hap[j-1]
  I[i][j] = ... that end with read[i-1] paired to a gap
  D[i][j] = ... that end with hap[j-1] paired to a gap
"""

from pathlib import Path
import yaml

DEFAULT_GLOBAL_PARAMS = Path(__file__).resolve().parent / "global_params.yaml"


class PairHMM:

    def __init__(self, read, hap, params_path=None):
        self.read = read
        self.hap = hap
        path = params_path or DEFAULT_GLOBAL_PARAMS
        (self.transition_matrix, self.p_match,
         self.p_mismatch, self.gap_emit) = self._naive_parameters(path)


#* --------------------------------------------------------------------------------
#* PARAMETER LOADING AND PREPARATION
#* --------------------------------------------------------------------------------

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


#* --------------------------------------------------------------------------------
#* FORWARD ALGORITHM
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
    total, M, I, D = PairHMM(read, hap).naive_forward()
    show("M (read base on haplotype base)", M, read, hap)
    show("I (read base on gap)", I, read, hap)
    show("D (haplotype base on gap)", D, read, hap)
    print(f"\n  forward total        = {total:.6f}")
