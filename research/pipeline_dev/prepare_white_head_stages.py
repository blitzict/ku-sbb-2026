"""Explicit dependency graph for the retained late weight-space readouts.

Run during package preparation, not during model fitting. Historical JSONs
are used only for recorded member names and file dependencies. No archived
row predictions, features, fitted coefficients or scores enter the rebuild.
"""
import ast
import json
from pathlib import Path
from pipeline.prepare_feature_stages import ROOT, code_closure
from pipeline.prepare_mechanism_stages import prior_outputs
from pipeline.prepare_white_model_stages import WS, SPLITS, tables, arrays, folds, candidate, build as specialists

SEEDS = (7, 29, 0, 123, 2024, 11041)
SCRIPT = "experiments/weight_space/"


def cached(name):
    return [WS + f"frozen_{name}/s{s}_f{f}.npz" for s in SEEDS for f in range(5)] + [WS + f"frozen_{name}/full.npz"]


def results(name, prefix=""):
    return [WS + prefix + name + ".json", WS + prefix + name + "_oof.npz", *candidate(name)]


def member_files(names):
    result = []
    for name in names:
        result += folds("physical_ft_8", ("fold{k}.npz",)) if name == "net" else [WS + name + ".npz"]
    return result


def original_inputs():
    """The explicit read_data/expand input closure, without model checkpoint bloat."""
    names = json.loads((ROOT / "submissions/sub_seqw5_stack.json").read_text())["members"]
    out = ["new weights/ensemble.json", "submissions/sub_seqw5_stack.json"]
    for n in names:
        stem = "tune_out/weight_edit_20261104/{part}_head_v2_bag.npy" if n == "MLPHV_bag_w" else "tune_out/{part}_" + n + ".npy"
        out += [stem.format(part=p) for p in ("oof", "test")]
    for family in ("VARC4hv_ns3", "VARC4_lm", "VARC4_full"):
        out += [f"tune_out/ver_{family}_s{s}.npz" for s in (0, 1)]
    out += [f"tune_out/weight_edit_20261104/ver_head_v2_{n}.npz" for n in ("MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2")]
    out += tables("lr5 gh lm ps", "tune_out/")
    # read_data verifies every recorded input hash even if not read again.
    out += list(json.loads((ROOT / "new weights/ensemble.json").read_text())["input_hashes"])
    return list(dict.fromkeys(x.replace("\\", "/") for x in out if not x.endswith(".py")))


def build():
    common = specialists()["stages"][0]["inputs"]
    original = original_inputs()
    guard = ["new weights/hard_state_tree_candidate/weights.json",
             "new weights/hard_state_tree_candidate/submission.csv",
             "new weights/hard_state_tree_candidate/replay.py",
             "tune_out/generator_segments.parquet", "tune_out/weight_edit_20261104/pruning_oof.npz"]
    loop = [*guard, WS + "white_loop_parent.npz"]
    priors = [p for s in SEEDS for p in prior_outputs(WS + f"law_outer{s}", priors=True)]
    stages = []

    def add(name, module, args, outputs, inputs=(), use_common=True, helper=False):
        script = ("pipeline_helpers/" if helper else SCRIPT) + module + ".py"
        stages.append(dict(id="white_head_" + name, argv=[script, *map(str, args)],
            inputs=list(dict.fromkeys([*(common if use_common else []), *inputs])),
            outputs=list(dict.fromkeys(outputs)), code_inputs=code_closure(script),
            env={"KU_SBB_THREADS": "2", "KU_SBB_REBUILD_FIXED": "1", "SAVE_WEIGHTS": "1"}))

    # The earlier head graph supplies the files read by er.parent/tables.
    early_file = ROOT / "pipeline/early_head_stages.json"
    early_outputs = []
    if early_file.exists():
        early_outputs = [o for stage in json.loads(early_file.read_text())["stages"] for o in stage["outputs"]
                         if not o.endswith((".pt", ".cbm", ".txt", ".ubj"))]
    early_outputs = [p for p in early_outputs if p != WS + "white_loop_parent.npz"]
    add("parent_cache", "white_loop", [], [WS + "white_loop_parent.npz"],
        [*guard, *original, *early_outputs, WS + "hard_state_tree_verified_oof.npz"])

    core = "net lawtree_both claude_law claude_jump lawshared lawblock lawconditional".split()
    branch = "physical_pulse_state_walk_" + "_".join(core) + "_rank"
    add("physical_reference", "white_extended", ["pulse", "state", "walk", *core, "--rank"],
        results(branch, "wn_"), [*loop, *priors, *member_files(core)])
    tag = "consult_full_bags_vol"
    add("consult", "white_consult", ["full", "--extra", "claude_bag2", "claude_bag3", "--tag", "bags_vol", "--physical", "volatility"],
        results(tag, "wn_"), [*loop, *priors, *member_files([*core, "claude_bag2", "claude_bag3"]), WS + "wn_" + branch + "_oof.npz"])
    extra = "claude_crop1 claude_allcrop1 claude_cropbag claude_two_crop claude_cropvol claude_cropvolbag claude_rev claude_revvol symtree_orig_c1_c2 symtree_extra_orig_c1_c2 lawvoltree lawpredictive lawforecast lawrank_original lawrank_crops".split()
    stage_inputs = [*loop, *priors, *member_files([*core, "claude_bag2", "claude_bag3", *extra]),
                    *tables("transition_odds component_moves_state"), WS + "wn_consult_full_bags_vol_oof.npz"]
    add("reversible_rank", "white_stage", ["reversible_rank", "--tables", "transition_odds", "component_moves_state", "--extra", *extra],
        results("stage_reversible_rank", "wn_"), stage_inputs)
    add("freeze_reversible_rank", "white_frozen_stage", ["reversible_rank"], cached("stage_reversible_rank"),
        [*stage_inputs, *results("stage_reversible_rank", "wn_")])

    tags = ["MECH3C12V_s0", "MECH3C12V_s1", "MECH3C12V_s2", "MECH3C12R_s0"]
    add("paired_views", "white_view_contrasts", ["paired_view_v1", *tags],
        [*tables("paired_view_v1"), WS + "paired_view_v1_sources.json"],
        ["tune_out/var/mat_ns3_plain.npz", *[f"tune_out/ver_{t}.npz" for t in tags]], use_common=False)
    agreement_members = "claude_cropvolbag claude_revlow claude_revorig lawcrop_c1_c2_rev lawrank_original lawrank_crops lawtree_both lawvoltree claude_law claude_jump lawshared lawconditional lawpredictive lawfourier lawblock lawforecast lawretrieval net".split()
    agreement_inputs = [*loop, *cached("stage_reversible_rank"), *candidate("stage_reversible_rank"),
        *tables("paired_view_v1 latent_level_pulse transport_state"), *member_files(agreement_members)]
    for name, module, args in (
        ("family_agreement", "white_expert_agreement", ["reversible_rank", "family_agreement"]),
        ("early_opinions", "white_baseline_opinions", ["early_opinions", "lawpatchforecast"]),
    ):
        tag = "local_rank_" + name
        needed = [*agreement_inputs, *original, *member_files(["lawpatchforecast"])] if name == "early_opinions" else agreement_inputs
        add(name, module, args, results(tag), needed)
        add("freeze_" + name, "white_freeze_agreement", [tag], cached(tag), [*needed, *results(tag)])

    seq = ["new weights/model_manifest.json"]
    for t, seed in (("W_SEQ_q", 0), ("W_SEQ_q_s1", 1), ("W_SEQ_qh", 0), ("W_SEQ_q_s2", 2), ("W_SEQ_qh_s1", 1)):
        seq += [f"new weights/models/{t}/fold{f}_seed{seed}.pt" for f in range(5)]
        seq += [f"tune_out/seqpart/{t}_f{f}.npz" for f in range(5)]
    for tag in ("mode_residual", "tail_channels"):
        add(tag + "_channels", "white_channel_replay", [tag],
            [*tables(tag + "_seq " + tag + "_channel_effect"),
             *[WS + f"{tag}_replay/fold{f}.npz" for f in range(5)],
             *member_files([f"law_{tag}_core", f"law_{tag}_other"])],
            [*seq, *arrays(tag + "_streams"), *tables(tag + " pulse_seq")])
    add("reliable_signals", "white_reliable_signals", [], tables("reliable_signals"),
        [*loop, *tables("mode_residual mode_residual_channel_effect tail_edges tail_channels tail_channels_channel_effect pulse_seq")])
    signal_inputs = [*loop, *cached("local_rank_early_opinions"), *candidate("local_rank_early_opinions"), *tables("reliable_signals")]
    audit_file = WS + "signal_ablation_local_rank_early_opinions_reliable_modes.json"
    add("mode_check", "white_signal_audit", ["local_rank_early_opinions", "reliable_modes"], [audit_file], signal_inputs)
    add("mode_peaks", "white_signal_package", ["local_rank_early_opinions", "early_mode_peaks", "reliable_modes"],
        results("local_rank_early_mode_peaks"), [*signal_inputs, audit_file])
    sources = ["claude_crop_three_partitions", "law_independent_rank_bag_29_7"]
    add("three_source_votes", "white_fold_vote", ["three_source_votes", "physical", *sources],
        [*results("local_rank_three_source_votes"), *cached("local_rank_early_mode_peaks")],
        [*signal_inputs, *results("local_rank_early_mode_peaks"), *member_files(sources)])

    upstream_members = "claude_crop_three_partitions law_independent_rank_bag_29_7 law_linked_rank_s29 claude_revorig claude_bag3 lawforecast lawblock lawshared lawretrieval lawvoltree lawpredictive net law_rank_dependence_s29".split()
    versions = [f"MECH3C12V_s{s}" for s in range(3)] + ["MECH3C12V_k29_s0", "MECH3C12V_k7_s0", "VARC4harm_ns3_s0", "VARC4harm_ns3_s1"]
    upstream_inputs = [*loop, *original, *member_files(upstream_members), *results("local_rank_three_source_votes"),
        "tune_out/var/mat_ns3_plain.npz", *[f"tune_out/ver_{t}.npz" for t in versions],
        *[f"tune_out/{p}_{t}.npy" for t in versions[:5] for p in ("oof", "test")]]
    upstream_outputs = [WS + "local_rank_upstream_joint.json", WS + "local_rank_upstream_joint_oof.npz"]
    add("upstream", "white_upstream_stack", ["--blend", ".25"], upstream_outputs, upstream_inputs)
    inner_outputs = [WS + f"upstream_inner/s{s}_f{f}.npz" for s in SEEDS for f in range(5)]
    add("upstream_inner", "fixed_final_readouts", ["upstream-inner"], inner_outputs,
        [*upstream_inputs, *upstream_outputs], helper=True)
    first = "local_rank_first_layer_overall"
    add("first_layer", "fixed_final_readouts", ["first-layer"], results(first),
        [*upstream_inputs, *upstream_outputs], helper=True)
    add("freeze_first_layer", "white_freeze_first_layer", [], cached(first),
        [*loop, *results(first), *results("local_rank_three_source_votes"), *upstream_outputs,
         *cached("local_rank_early_mode_peaks"), *member_files(sources), *inner_outputs])
    tree = ast.parse((ROOT / "pipeline/legacy_cpu/experiments/weight_space/white_opinion_spectrum.py").read_text())
    families = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAMILIES" for t in n.targets))
    paired = "law_ordered_paths law_path_control law_shock_quantiles law_shock_quantile_control law_rank_dependence_s7 law_rank_dependence_s29 claude_masked_physics claude_source_loss".split()
    add("shared_response", "fixed_final_readouts", ["shared-response"], results("local_rank_shared_residual_response"),
        [*loop, *original, *cached(first), *candidate(first), *tables("mode_residual"),
         *member_files([*[n for family in families for n in family], *paired])], helper=True)
    return dict(status="Retained late graph; full runtime validation pending", stages=stages,
                omitted_unselected_branches=["concordance recovery route", "control-rank calibration", "family ablation search", "agreement candidate", "extra diagnostic partition 61817"],
                equivalence_evidence="pipeline/retained_branch_equivalence.json")


if __name__ == "__main__":
    value = build()
    path = ROOT / "pipeline/white_head_stages.json"
    path.write_text(json.dumps(value, indent=2))
    print(f"Wrote {len(value['stages'])} late readout stages to {path}")
