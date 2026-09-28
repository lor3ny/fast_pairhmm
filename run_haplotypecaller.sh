#!/usr/bin/env bash
# End-to-end GATK HaplotypeCaller demo on synthetic data, plus a check of
# gatk_forward() in fast_forward.py against GATK's own PairHMM
# (done by compare_with_gatk.py).
# Requires: gatk, bwa, samtools (all on PATH),
#           python3 with PyYAML (Fedora: sudo dnf install python3-pyyaml)
#
# Folder layout:
#   synthetic_data/     simulated inputs: ref.fa, reads_R1.fq, reads_R2.fq, truth.tsv
#   intermediate_data/  reference copy + bwa index, .fai, .dict, sorted BAM + .bai,
#                       PairHMM dump + the debug VCF from the run that wrote it
#   output/             VCF, GVCF (+ .tbi indexes), called.tsv, comparison.tsv,
#                       pairhmm_ours.tsv, pairhmm_comparison.tsv
set -euo pipefail

DATA="synthetic_data"
INTER="intermediate_data"
OUT="output"



#! CHECKS
for script in "$DATA/simulate_reads.py" fast_forward.py compare_with_gatk.py; do
  [[ -f "$script" ]] || { printf "ERROR: $script not found\n" >&2; exit 1; }
done
mkdir -p "$DATA" "$INTER" "$OUT"
#! CHECKS


#! PREPARING INPUT FOR VARIANT CALLING: SYNTETHETIC DATA GENERATION, ALIGNEMENT, INDEXING, SORTING

printf "=================== 1. Simulate reference + reads into $DATA/ ===================\n"
# The Python script writes to the current directory, so run it from inside $DATA
(cd "$DATA" && python3 simulate_reads.py)
printf "\n\n\n"



printf "=================== 2. Index reference into $INTER/ (bwa, faidx, sequence dictionary) ===================\n"
# GATK looks for ref.fa.fai and ref.dict NEXT TO the reference it is given,
# so the indexes and the reference must share a folder. We copy the reference
# into $INTER and use that copy for all downstream steps.
cp "$DATA/ref.fa" "$INTER/ref.fa"
REF="$INTER/ref.fa"
bwa index "$REF" 2>/dev/null
samtools faidx "$REF"
rm -f "$INTER/ref.dict"
gatk CreateSequenceDictionary -R "$REF" -O "$INTER/ref.dict"
printf "\n\n\n"



printf "=================== 3. Align, sort, index into $INTER/ (read group is required by GATK) ===================\n"
bwa mem -t 4 -R '@RG\tID:rg1\tSM:sample1\tPL:ILLUMINA\tLB:lib1' \
    "$REF" "$DATA/reads_R1.fq" "$DATA/reads_R2.fq" 2>/dev/null \
  | samtools sort -o "$INTER/sample1.bam" -
samtools index "$INTER/sample1.bam"
printf "\n\n\n"



#! THE FOLLOWING VERSIONS ARE COMMENTED OUT, BECAUSE THEY ARE NOT NEEDED FOR THE PAIRHMM DEBUG RUN. THEY ARE KEPT HERE FOR REFERENCE.
#! THEY ARE MORE OPTIMIZED! SO KEEP THEM FOR THE FUTURE

# printf "=================== 4. HaplotypeCaller: regular VCF into $OUT/ ===================\n"
# gatk HaplotypeCaller -R "$REF" -I "$INTER/sample1.bam" -O "$OUT/sample1.vcf.gz"
# printf "\n\n\n"



# printf "=================== 5. HaplotypeCaller: GVCF mode into $OUT/ (for joint genotyping later) ===================\n"
# gatk HaplotypeCaller -R "$REF" -I "$INTER/sample1.bam" -O "$OUT/sample1.g.vcf.gz" -ERC GVCF
# printf "\n\n\n"



# printf "=================== 6. Compare calls to truth ==================="
# zcat "$OUT/sample1.vcf.gz" | grep -v '^#' \
#   | awk -v OFS='\t' '{split($10,g,":"); print $2,$4,$5,g[1]}' > "$OUT/called.tsv"
# {
#   printf "POS\tREF\tALT\tTRUTH\tCALLED_GT\n"
#   join -t $'\t' -a1 -e MISSED -o 1.1,1.2,1.3,1.4,2.4 \
#       <(sort -k1,1 "$DATA/truth.tsv") <(sort -k1,1 "$OUT/called.tsv") | sort -n
# } > "$OUT/comparison.tsv"
# column -t "$OUT/comparison.tsv"
# printf "Truth variants: $(wc -l < "$DATA/truth.tsv")   Called variants: $(wc -l < "$OUT/called.tsv")"
# printf "\n\n\n"


#! JAVA GATK FORWARD

printf "=================== 7. HaplotypeCaller debug run: dump every PairHMM input and result ===================\n"
# Default: the whole 20 kb genome (~2,600 read/haplotype pairs, a few seconds
# for gatk_forward()). With real data, restrict it to a few regions, e.g.
#   PAIRHMM_REGIONS="chr1:1150-1250 chr1:2350-2450" bash run_haplotypecaller.sh
L_ARGS=()
for region in ${PAIRHMM_REGIONS:-}; do L_ARGS+=(-L "$region"); done
printf "Regions: ${PAIRHMM_REGIONS:-whole genome}"
# LOGLESS_CACHING is GATK's pure-Java, double-precision PairHMM: the same
# arithmetic as gatk_forward(). The default AVX version starts in single
# precision, so its numbers can differ slightly in the last digits.
gatk HaplotypeCaller -R "$REF" -I "$INTER/sample1.bam" \
    -O "$INTER/pairhmm_debug.vcf.gz" ${L_ARGS[@]+"${L_ARGS[@]}"} \
    --pair-hmm-implementation LOGLESS_CACHING \
    --pair-hmm-results-file "$INTER/pairhmm_dump.txt"
printf "\n\n\n"


#! CUSTOM FAST FORWARD

printf "=================== 8. Recompute every PairHMM likelihood with gatk_forward() ===================\n"
python3 fast_forward.py --dump "$INTER/pairhmm_dump.txt" \
    --out "$OUT/pairhmm_ours.tsv"
printf "\n\n\n"



#! COMPARE

printf "=================== 9. Compare our likelihoods with GATK's ===================\n"
status=0
python3 compare_with_gatk.py "$INTER/pairhmm_dump.txt" "$OUT/pairhmm_ours.tsv" \
    --out "$OUT/pairhmm_comparison.tsv" || status=$?
printf "\n\n\n"



printf "Results written to $OUT/"
exit $status
