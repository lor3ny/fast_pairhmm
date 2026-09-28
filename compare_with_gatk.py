"""
Compare the likelihoods computed by naive_forward.py with GATK's own.

Two separate phases:

  1. python3 naive_forward.py --dump pairhmm_dump.txt --out ours.tsv
         runs gatk_forward() on every read/haplotype pair of the dump.

  2. python3 compare_with_gatk.py pairhmm_dump.txt ours.tsv --out comparison.tsv
         reads GATK's value for each pair from the dump and ours from
         ours.tsv, and reports the differences (see compare_with_gatk()).

The dump (from `gatk HaplotypeCaller --pair-hmm-results-file`) holds one
line per read/haplotype pair: the exact bases and per-base qualities GATK
fed to its PairHMM, plus GATK's result.
"""

import argparse
import sys


def read_gatk_results(path):
    """Read GATK's log10 P(read|hap) for every pair of a PairHMM dump.

    GATK writes one header line,
        # hap-bases read-bases read-qual read-ins-qual read-del-qual gcp expected-result
    then one line per read/haplotype pair with those 7 space-separated
    fields. Only the last one, GATK's result, is used here.

    Returns {line_number: (gatk_log10, gatk_text)}, where gatk_text is
    the number exactly as GATK printed it."""
    results = {}
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"{path}:{line_no}: expected 7 fields, found {len(fields)}")
            # Accept a comma decimal separator too ("-5,156349e+00"), in
            # case the JVM formats numbers with a non-English locale.
            gatk_text = fields[6].replace(",", ".")
            results[line_no] = (float(gatk_text), gatk_text)
    return results


def read_our_results(path):
    """Read the TSV written by `naive_forward.py --dump ... --out ...`.
    Yields (dump_line, read_len, hap_len, log10_likelihood)."""
    with open(path) as f:
        next(f)  # header
        for line in f:
            dump_line, read_len, hap_len, ll = line.split("\t")
            yield int(dump_line), int(read_len), int(hap_len), float(ll)


def compare_with_gatk(dump_path, ours_path, out_path=None, rel_tol=1e-6):
    """Compare every likelihood in ours_path with GATK's value for the
    same dump line.

    Writes one row per pair to out_path (if given), prints a summary and
    returns 0 if every pair agrees within rel_tol, else 1.

    GATK prints its value with only 7 significant digits (Java's "%e"),
    so even a perfect re-implementation can differ by up to ~5e-7 of the
    value. The default tolerance of 1e-6 allows for that rounding only.
    As a stricter check, each result is also printed the way GATK prints
    it and compared digit by digit (column same_digits)."""
    gatk = read_gatk_results(dump_path)
    rows = []
    for line_no, read_len, hap_len, ours in read_our_results(ours_path):
        if line_no not in gatk:
            raise ValueError(f"{ours_path}: dump line {line_no} is not a pair in {dump_path}")
        gatk_ll, gatk_text = gatk[line_no]
        abs_diff = abs(ours - gatk_ll)
        rel_diff = abs_diff / abs(gatk_ll) if gatk_ll != 0.0 else abs_diff
        status = "OK" if rel_diff <= rel_tol else "DIFF"
        same_digits = "yes" if f"{ours:e}" == gatk_text else "no"
        rows.append((line_no, read_len, hap_len, gatk_ll, ours, abs_diff, rel_diff,
                     same_digits, status))

    if not rows:
        print(f"No read/haplotype pairs found in {ours_path}")
        return 1

    if out_path:
        with open(out_path, "w") as f:
            f.write("dump_line\tread_len\thap_len\tgatk_log10\tours_log10"
                    "\tabs_diff\trel_diff\tsame_digits\tstatus\n")
            for r in rows:
                f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6e}\t{r[4]:.10e}"
                        f"\t{r[5]:.2e}\t{r[6]:.2e}\t{r[7]}\t{r[8]}\n")

    n_diff = sum(r[8] == "DIFF" for r in rows)
    n_same = sum(r[7] == "yes" for r in rows)
    worst = max(rows, key=lambda r: r[6])
    read_lens = [r[1] for r in rows]
    hap_lens = [r[2] for r in rows]
    print(f"gatk_forward() vs GATK PairHMM: {len(rows)} of {len(gatk)} read/haplotype pairs")
    print(f"  read lengths {min(read_lens)}-{max(read_lens)}, "
          f"haplotype lengths {min(hap_lens)}-{max(hap_lens)}")
    print(f"  largest |ours - GATK|    : {max(r[5] for r in rows):.2e} (log10 units)")
    print(f"  largest relative diff    : {worst[6]:.2e} at dump line {worst[0]} "
          f"(tolerance {rel_tol:.0e})")
    print(f"  identical to all 7 digits GATK prints: {n_same}/{len(rows)}")
    print(f"  pairs outside tolerance  : {n_diff}")
    if out_path:
        print(f"  per-pair table           : {out_path}")
    if n_diff == 0:
        print("  RESULT: PASS -- gatk_forward() reproduces GATK's likelihoods")
        return 0
    print("  RESULT: FAIL -- see the rows marked DIFF")
    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare the likelihoods written by naive_forward.py --dump "
                    "with GATK's own values from the same PairHMM dump.")
    parser.add_argument("dump",
                        help="file written by gatk HaplotypeCaller --pair-hmm-results-file")
    parser.add_argument("ours",
                        help="TSV written by naive_forward.py --dump DUMP --out TSV")
    parser.add_argument("--out", metavar="TSV",
                        help="write the per-pair comparison table here")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="largest relative difference counted as agreement "
                             "(default 1e-6)")
    args = parser.parse_args()

    sys.exit(compare_with_gatk(args.dump, args.ours, args.out, args.tol))
