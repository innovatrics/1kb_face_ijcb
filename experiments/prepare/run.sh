#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Data preparation for the paper (Section 3 and the crop sets of Section 8), starting
# from the aligned crops under $FACE1KB_DATA_ROOT/<dataset>/aligned_<res>/.
#
#   1. index.csv and pairs.parquet of both datasets         CPU, ~1 min, ~3 GB RAM
#   2. Color FERET labels.csv from the NIST ground truth    CPU, ~1 min
#   3. AI-Solutions-KK attributes.csv (age, gender, MST)    CPU, ~1 min
#   4. crop-tightness variants aligned_112_{tight,mid,fill} CPU, ~1 min
#      and aligned_56 of both datasets
#   5. preprocessed Color FERET crops aligned_112_<op>      A1-A4 CPU, a few min;
#                                                           B1/B2/C1/C2 1 GPU, ~25 min
#   6. dataset statistics, dataset figures, Color FERET attribute table   CPU, seconds
#
# Environment:
#   NIST     NIST Color FERET distribution (extracted folder or the tar archive);
#            required for step 2
#   KK_RAW   raw AI-Solutions-KK download (one folder per identity); optional, for
#            the source file-size and dimension figures of step 6
#   PYTHON   interpreter (default: python)
#   SKIP     space-separated step numbers to skip, e.g. SKIP="5"
#   DATASETS datasets to prepare (default: "colorferet kk"); a dataset without
#            aligned_112 or aligned_224 crops is skipped with a notice, so a copy of
#            only the AI-Solutions-KK crops (or only your Color FERET alignment)
#            works without setting it
#   REDERIVE_112
#            set to 1 to replace an existing aligned_112 by aligned_224[::2, ::2]
#            (for crop deliveries whose 112 px crops come from another alignment;
#            build_index.py warns about those)
#
# Steps 1-3 skip output files that exist (index, pairs, labels, attributes); steps 4
# and 5 resume, writing only missing crops. Step 6 regenerates its aggregate
# summaries, figures and table on every run.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
SKIP="${SKIP:-}"
DATASETS="${DATASETS:-colorferet kk}"

# path of a face1kb.config helper, e.g. cfg 'aligned_dir("kk", 112)'
cfg() { "$PY" -c "from face1kb import config; print(config.$1)"; }

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}
# has_res DS RES: the aligned_<RES> crop folder of DS exists
has_res() { [ -d "$(cfg "aligned_dir('$1', $2)")" ]; }
# selected DS: DS is in $DATASETS and has aligned_112 or aligned_224 crops
selected() {
	[[ " $DATASETS " == *" $1 "* ]] || return 1
	has_res "$1" 112 || has_res "$1" 224
}

active=""
for ds in $DATASETS; do
	if selected "$ds"; then
		active="$active $ds"
	else
		echo "no aligned_112 or aligned_224 crops of '$ds' under" \
			"$(cfg "dataset_dir('$ds')"): skipping it" >&2
	fi
done

# ---------------------------------------------------------------- 1. index + pairs
if ! skip 1; then
	for ds in $active; do
		# a 224 px-only crop release: aligned_112 is exactly aligned_224[::2, ::2]
		if has_res "$ds" 224; then
			if ! has_res "$ds" 112; then
				run experiments/prepare/make_crop_variants.py --dataset "$ds" \
					--variants "" --resolutions 112
			elif [ "${REDERIVE_112:-0}" = 1 ]; then
				run experiments/prepare/make_crop_variants.py --dataset "$ds" \
					--variants "" --resolutions 112 --overwrite
			fi
		fi
		have=""
		for r in 64 96 112 168 224; do
			if has_res "$ds" "$r"; then
				have="$have${have:+,}$r"
			fi
		done
		[ -f "$(cfg "index_csv('$ds')")" ] ||
			run experiments/prepare/build_index.py --dataset "$ds" --check-res "$have"
		[ -f "$(cfg "pairs_parquet('$ds')")" ] ||
			run experiments/prepare/define_pairs.py --dataset "$ds"
	done
fi

# ---------------------------------------------------------------- 2. CF labels
if ! skip 2 && [ ! -f "$(cfg 'labels_csv()')" ]; then
	if [ -z "${NIST:-}" ]; then
		echo "NIST is not set: skipping the Color FERET labels (step 2)" >&2
	else
		run experiments/prepare/prepare_colorferet_labels.py --nist "$NIST"
	fi
fi

# ---------------------------------------------------------------- 3. KK attributes
# genderage runs on the CPU (the default), which reproduces the paper's genderage
# outputs bit for bit given the same crops (see docs/datasets.md, Known deviations)
if ! skip 3 && selected kk && [ ! -f "$(cfg 'attributes_csv()')" ]; then
	if "$PY" -c 'import stone' 2>/dev/null; then
		run experiments/prepare/estimate_attributes.py --dataset kk
	else
		echo "skin-tone-classifier (stone, GPL-3.0) is not installed: estimating" \
			"age and gender only" >&2
		run experiments/prepare/estimate_attributes.py --dataset kk --no-mst
	fi
fi

# ---------------------------------------------------------------- 4. crop sets
# always invoked: existing crops are kept, so an interrupted run is completed
if ! skip 4; then
	if selected colorferet && has_res colorferet 224; then
		run experiments/prepare/make_crop_variants.py --dataset colorferet
	fi
	for ds in $active; do
		run experiments/prepare/make_crop_variants.py --dataset "$ds" \
			--variants "" --resolutions 56
	done
fi

# ---------------------------------------------------------------- 5. preprocessing
if ! skip 5 && selected colorferet; then
	# resumable: existing outputs are skipped
	run experiments/prepare/preprocess.py --dataset colorferet
fi

# ---------------------------------------------------------------- 6. report
if ! skip 6; then
	if selected kk && [ -f "$(cfg "index_csv('kk')")" ]; then
		if [ -n "${KK_RAW:-}" ]; then
			run experiments/prepare/dataset_report.py --dataset kk --raw-dir "$KK_RAW"
		else
			run experiments/prepare/dataset_report.py --dataset kk
		fi
	fi
	if selected colorferet && [ -f "$(cfg "index_csv('colorferet')")" ]; then
		run experiments/prepare/dataset_report.py --dataset colorferet
	fi
	if [ -f "$(cfg 'labels_csv()')" ]; then
		run experiments/prepare/cf_attribute_table.py
	fi
fi
echo "done"
