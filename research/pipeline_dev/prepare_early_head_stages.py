"""Early head chain and early feature producers, as executable stages for pipeline.run_stages.

Heads, expert_fusion through hard_state_tree, take their inputs from the hashes each historical run recorded in its weights.json and
validation.json. Producer inputs are listed explicitly from each script's read statements, including names built in loops. Head outputs are
declared only when the historical run wrote a file of that name, so a stage cannot be marked done with a misnamed output.
Owner: Claude. Settings are the fixed historical recipes; no new search.
"""
import argparse
import json
import re
from pathlib import Path

from pipeline.prepare_feature_stages import ROOT, code_closure
from pipeline.prepare_rebuild_stages import order

WS = "tune_out/weight_space"
PARTS = ("train", "test")
ENV = {"KU_SBB_THREADS": "2", "KU_SBB_REBUILD_FIXED": "1", "SAVE_WEIGHTS": "1"}
COMMON = ["data/train_labels.csv", "clean/features_train.parquet", "clean/features_test.parquet"]
EXT = "extended_curve_net_bank_both_pulse_joint_head_family_state_net"
CAND = lambda folder, *names: [f"new weights/{folder}/{n}" for n in names]
HEADS = [  # stage id, script and arguments, recorded-input folder, parent oof, outputs written by the fixed source
    ("early_head_1_expert_fusion", ["expert_fusion.py"], "expert_fusion_candidate", None,
     CAND("expert_fusion_candidate", "weights.json", "submission.csv") + [f"{WS}/expert_fusion_oof.npz", f"{WS}/expert_fusion_results.json"]),
    ("early_head_2_residual_harmonic", ["residual_trial.py", "--tag", "harmonic"], "harmonic_candidate", "expert_fusion",
     CAND("harmonic_candidate", "weights.json", "submission.csv") + [f"{WS}/harmonic_oof.npz", f"{WS}/harmonic_results.json"]),
    ("early_head_3_seasonal_sequence", ["seasonal_head.py", "--sequence"], "seasonal_seq_candidate", "harmonic",
     CAND("seasonal_seq_candidate", "weights.json", "submission.csv") + [f"{WS}/seasonal_seq_oof.npz", f"{WS}/seasonal_seq_results.json"]),
    ("early_head_4_mechanism_phase_ar_mix_transfer", ["mechanism_edit.py", "--mode", "phase_ar_mix_transfer"], "mechanism_phase_ar_mix_transfer_candidate", "seasonal_seq",
     CAND("mechanism_phase_ar_mix_transfer_candidate", "weights.json", "submission.csv") + [f"{WS}/mechanism_phase_ar_mix_transfer_oof.npz", f"{WS}/mechanism_phase_ar_mix_transfer_results.json"]),
    ("early_head_5_trend_net", ["trend_870.py", "--mode", "net"], "trend870_net_candidate", "mechanism_phase_ar_mix_transfer",
     CAND("trend870_net_candidate", "weights.json", "submission.csv", "replay_inputs.npz", "validation.json") + [f"{WS}/trend870_net_oof.npz", f"{WS}/trend870_net_results.json"]),
    ("early_head_6_extend_readout", ["extend_readout.py", "--spec", "curve.net+bank.both+pulse.joint+head.family+state.net"], f"{EXT}_candidate", "trend870_net",
     CAND(f"{EXT}_candidate", "weights.json", "submission.csv", "replay_inputs.npz", "validation.json", "replay.py", "submission_replay.csv") + [f"{WS}/{EXT}_oof.npz", f"{WS}/{EXT}_results.json"]),
    ("early_head_7_hard_state_tree", ["hard_expert_trial.py", "--kind", "state_tree"], "hard_state_tree_candidate", EXT,
     [f"{WS}/hard_state_tree.json", f"{WS}/hard_state_tree_oof.npz"]),
    ("early_head_8_hard_state_tree_verified", ["hard_expert_trial.py", "--kind", "state_tree", "--verify"], "hard_state_tree_candidate", EXT,
     CAND("hard_state_tree_candidate", "weights.json", "submission.csv", "replay_inputs.npz", "validation.json", "replay.py", "submission_replay.csv") + [f"{WS}/hard_state_tree_verified.json", f"{WS}/hard_state_tree_verified_oof.npz"])]
# Every head starts with expert_fusion.read_data(): the stage 7 ensemble, its member list, all 51 member predictions and the edited MLP bag.
MEMBERS = json.loads((ROOT / "pipeline/legacy_cpu/model_manifest.json").read_text())
MEMBERS = json.loads(next(s for s in MEMBERS["steps"] if s["argv"][1] == "WRITE_JSON")["argv"][3])["members"]
READ_DATA = ["new weights/ensemble.json", "submissions/sub_seqw5_stack.json"] + \
    [f"tune_out/{p}_{m}.npy" for m in MEMBERS if m != "MLPHV_bag_w" for p in ("oof", "test")] + \
    [f"tune_out/weight_edit_20261104/{p}_head_v2_bag.npy" for p in ("oof", "test")] + \
    [f"tune_out/weight_edit_20261104/ver_head_v2_{t}.npz" for t in ("MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2")]
HARD_EXTRA = ["tune_out/weight_edit_20261104/pruning_oof.npz", "tune_out/generator_segments.parquet", f"{WS}/state_train.parquet", f"{WS}/state_test.parquet"]
S = lambda *names: [f"{WS}/{n}_{p}.{'npz' if n.endswith('_streams') else 'parquet'}" for n in names for p in PARTS]
T = lambda *names: [f"tune_out/{n}_{p}.parquet" for n in names for p in PARTS]
HELP = "../../pipeline_helpers/early_feature_extract.py"
BANK = ["curve", "trend", "phase_law", "periodic", "mixture_law", "ar_law", "innov_v2", "noise_tests"]
PRODUCERS = [  # stage id, script and arguments, explicit outputs, explicit inputs (from each script's read statements)
    ("early_feat_trend_clock", ["trend_clock.py"], S("clock") + S("clock_streams"), S("pulse_streams", "periodic", "trend", "innov_v2")),
    ("early_feat_state_tests", ["state_tests.py"], S("state_tests"), S("pulse_streams", "state_streams")),
    ("early_feat_shape_tests", ["shape_bank.py", "prepare"], S("shape_tests"), S("curve_streams", "periodic_streams", "periodic")),
    ("early_feat_shape_counts", [HELP, "shape_counts"], [f"{WS}/shape_counts_{p}.npz" for p in PARTS], S("pulse_streams")),
    ("early_feat_deconv_pulse_measure", [HELP, "deconv_pulse_measure"], [f"{WS}/deconv_pulse_measure_{p}.npz" for p in PARTS], S("pulse_streams")),
    ("early_feat_conditional_scores", [HELP, "conditional_scores"], S("conditional_scores"), S("pulse_streams", "state_streams")),
    ("early_feat_trend_population", [HELP, "trend_population"], S("trend_population"), S("pulse_streams", "state")),
    ("early_model_transfer_member", ["transfer_member.py"], S("transfer_member") + [f"{WS}/transfer_model/{m}_fold{k}.txt" for m in ("noise", "pooled", "stationary") for k in range(5)], S("noise_tests", "innov_v2", "periodic"))]
for mode, extra in (("base", []), ("adjusted", ["shape_tests"]), ("state", ["shape_tests", "pulse", "state", "clock", "state_tests"])):
    PRODUCERS.append((f"early_model_shape_bank_{mode}", ["shape_bank.py", mode], S(f"shape_bank_{mode}") + [f"{WS}/shape_bank_{mode}/columns.json"] + [f"{WS}/shape_bank_{mode}/fold{k}.txt" for k in range(5)], S(*BANK, *extra) + T("lm", "gh", "lr5", "sn")))
SEQ_TAGS = {"W_SEQ_q": 0, "W_SEQ_q_s1": 1, "W_SEQ_q_s2": 2, "W_SEQ_qh": 0, "W_SEQ_qh_s1": 1}
SEQ_INPUTS = ["tune_out/seq_cache_train.npz", "tune_out/seq_cache_test.npz", "new weights/model_manifest.json"] + \
    [f"new weights/models/{t}/fold{k}_seed{s}.pt" for t, s in SEQ_TAGS.items() for k in range(5)] + [f"tune_out/seqpart/{t}_f{k}.npz" for t in SEQ_TAGS for k in range(5)]
for prefix in ("periodic", "innov_v2", "curve", "pulse", "state", "trend"):
    script = ["periodic_replay.py", "--prefix", prefix] if prefix != "trend" else ["../../pipeline_helpers/replay_trend_sequence.py"]
    PRODUCERS.append((f"early_seq_replay_{prefix}", script, [f"{WS}/{prefix}_seq_{p}.parquet" for p in PARTS] + [f"{WS}/{prefix}_replay/fold{k}.npz" for k in range(5)],
                      S(f"{prefix}_streams") + SEQ_INPUTS + (S("periodic") if prefix == "periodic" else [])))


def recorded(folder):
    """Every non-code file whose hash the historical candidate recorded."""
    h = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) and ("/" in k or "\\" in k): h[k.replace("\\", "/")] = v
                else: walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    for name in ("weights.json", "validation.json"):
        p = ROOT / "new weights" / folder / name
        if p.exists(): walk(json.loads(p.read_text()))
    return sorted(k for k in h if not k.endswith(".py"))


def build():
    stages = []
    for ident, argv, outs, reads in PRODUCERS:
        script = argv[0]
        path = ("pipeline_helpers/" + script.split("/")[-1]) if script.startswith("../") else f"experiments/weight_space/{script}"
        reads = [r for r in reads if r not in outs]
        stages.append(dict(id=ident, argv=[path, *argv[1:]], inputs=list(dict.fromkeys([*COMMON, *reads])), outputs=outs,
                           code_inputs=code_closure(path), env=ENV, evidence="explicit inputs from the script read statements; outputs named by source"))
    for ident, argv, folder, parent, outs in HEADS:
        ins = [i for i in recorded(folder) if not i.startswith(f"new weights/{folder}/")] + READ_DATA
        if parent: ins.append(f"{WS}/{parent}_oof.npz")
        if folder == "hard_state_tree_candidate": ins += HARD_EXTRA
        path = f"experiments/weight_space/{argv[0]}"
        stages.append(dict(id=ident, argv=[path, *argv[1:]], inputs=list(dict.fromkeys([*COMMON, *ins])), outputs=outs,
                           code_inputs=code_closure(path), env=ENV, evidence=f"inputs recorded by the historical {folder} run, plus read_data and parent archive; outputs from the fixed source"))
    available = set()
    for f in (ROOT / "pipeline").glob("*stages*.json"):
        if f.name == "early_head_stages.json": continue
        d = json.loads(f.read_text())
        available.update(o for s in d.get("stages", []) for o in s.get("outputs", []))
        available.update(d.get("prerequisite_inputs", []))
    clash = sorted({o for s in stages for o in s["outputs"]} & available)
    if clash: raise ValueError(f"Outputs already produced by root stages: {clash}")
    return dict(status="Early heads and producers; head inputs as recorded, producer inputs explicit; not yet run at full size",
                stages=order(stages, available), prerequisite_inputs=sorted(available))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "pipeline/early_head_stages.json")
    a = p.parse_args()
    result = build()
    a.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result['stages'])} early stages")
