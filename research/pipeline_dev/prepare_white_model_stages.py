"""Fixed specialist training recipes used by the final weight-space readout.

These are model fits, not an Optuna or architecture search. Early readouts and
the family-agreement cache are external dependencies declared below. The full
assembler must order these stages together with those head stages.
"""
import json
from pipeline.prepare_feature_stages import ROOT, code_closure
from pipeline.prepare_mechanism_stages import prior_outputs

WS = "tune_out/weight_space/"
SPLITS = ("train", "test")


def tables(names, folder=WS):
    return [f"{folder}{name}_{part}.parquet" for name in names.split() for part in SPLITS]


def arrays(names, folder=WS):
    return [f"{folder}{name}_{part}.npz" for name in names.split() for part in SPLITS]


def folds(folder, patterns=("fold{k}.pt", "fold{k}.npz"), count=5):
    return [WS + folder + "/" + pattern.format(k=k) for k in range(count) for pattern in patterns]


def candidate(name):
    return [f"new weights/wn_{name}_candidate/{file}" for file in
            ("weights.json", "submission.csv", "replay_inputs.npz", "validation.json", "replay.py")]


def build():
    common = ["data/train_labels.csv", "data/sample_submission.csv",
              *[f"clean/features_{p}.parquet" for p in SPLITS],
              *[f"tune_out/seq_cache_{p}.npz" for p in SPLITS]]
    common += tables("ar_law bounds curve phase_law trend periodic noise_tests innov_v2 mixture_law "
                     "state pulse volatility_law vol_smoothing walk_jump shape_tests state_tests clock "
                     "conditional_scores trend_population memory_law1 memory_law2 boundary_law_state "
                     "boundary_law_pulse hetero_jump filtered_step markov_copula component_moves_state "
                     "component_moves_pulse native_law_state native_law_pulse")
    common += tables(" ".join("jump_law_" + s for s in ("pulse", "state", "nullstate", "prestate", "smoothvol")))
    common += arrays("pulse_streams state_streams smoothvol_streams hetero_path shape_counts "
                     "deconv_pulse_measure wnlaw_physical")
    common += [f"tune_out/{name}_{p}.parquet" for name in ("lm", "gh", "lr5", "sn") for p in SPLITS]
    common += ["tune_out/var/mat_ns3_plain.npz", "tune_out/var/mat_ns3_c7.npz"]
    common += prior_outputs(WS + "law_outer0", priors=True)
    guard = [WS + "white_loop_parent.npz", "new weights/hard_state_tree_candidate/weights.json",
             "tune_out/generator_segments.parquet", "tune_out/weight_edit_20261104/pruning_oof.npz"]
    view_inputs = []
    for v in ("c1", "c2", "rev"):
        folder = WS + f"view_{v}/"
        view_inputs += tables("ar_law bounds curve phase_law trend periodic noise_tests innov_v2 mixture_law "
                              "state volatility_law walk_jump native_law_state "
                              "jump_law_pulse jump_law_state jump_law_nullstate jump_law_prestate jump_law_smoothvol", folder)
        view_inputs += prior_outputs(folder + "law_outer0")
    stages = []

    def add(name, module, args, outputs, extra=(), common_inputs=True):
        script = f"experiments/weight_space/{module}.py"
        stages.append(dict(id="white_model_" + name, argv=[script, *map(str, args)],
                           inputs=list(dict.fromkeys([*(common if common_inputs else []), *extra])),
                           outputs=list(dict.fromkeys(outputs)), code_inputs=code_closure(script),
                           env={"KU_SBB_THREADS": "2", "KU_SBB_REBUILD_FIXED": "1", "SAVE_WEIGHTS": "1"}))

    add("lawtree_both", "white_mech_tree", ["both"],
        [WS + "lawtree_both.npz", WS + "lawtree_both_models/columns.json", *folds("lawtree_both_models", ("fold{k}.txt",))])
    sequence = ["new weights/model_manifest.json"]
    sequence += [f"new weights/models/W_SEQ_q/fold{k}_seed0.pt" for k in range(5)]
    add("physical_ft8", "white_physical_net", ["--epochs", 8], folds("physical_ft_8"), [*guard, *sequence])

    # This historical intermediate supplies residual offsets to the law trees.
    branch = "physical_pulse_state_walk_net_lawtree_both_claude_law_rank"
    priors = [p for s in (7, 29, 123, 2024, 11041) for p in prior_outputs(WS + f"law_outer{s}", priors=True)]
    branch_inputs = [*guard, *priors, *folds("physical_ft_8"), WS + "lawtree_both.npz", WS + "claude_law.npz"]
    add("physical_parent", "white_extended", ["pulse", "state", "walk", "net", "lawtree_both", "claude_law", "--rank"],
        [WS + f"wn_{branch}.json", WS + f"wn_{branch}_oof.npz", *candidate(branch)], branch_inputs)
    add("frozen_physical_parent", "white_checkpoint", [], [WS + "frozen874_offsets.npz"],
        [*branch_inputs, WS + f"wn_{branch}_oof.npz", *candidate(branch)])
    add("lawvoltree", "white_second_tree", ["lawvoltree"],
        [WS + "lawvoltree.npz", WS + "lawvoltree_res.npz",
         *folds("lawvoltree_models", ("standalone{k}.txt", "residual{k}.txt"))], [*guard, WS + "frozen874_offsets.npz"])

    for extra in (False, True):
        tag = "symtree_" + ("extra_" if extra else "") + "orig_c1_c2"
        add(tag, "white_symmetric_tree", ["--views", "c1", "c2", *(["--extras"] if extra else [])],
            [WS + tag + ".npz", *folds(tag + "_models", ("fold{k}.cbm", "fold{k}.npz"))], view_inputs)
    for mode in ("original", "crops"):
        tag = "lawrank_" + mode
        add(tag, "white_rank_boost", [mode], [WS + tag + ".npz", *folds(tag + "_models", ("fold{k}.txt",))], view_inputs)
    for seeds, linked, count in (([29, 7], False, 5), ([29], True, 5), ([43], True, 10)):
        outputs, required = [], [*view_inputs, WS + "lawrank_crops.npz"]
        for seed in seeds:
            part = str(seed) if count == 5 else f"{count}x{seed}"
            tag = ("law_linked_rank_s" if linked else "law_independent_rank_s") + part
            outputs += [WS + tag + ".npz", *folds(tag + "_models", ("fold{k}.txt",), count)]
            for v in ("", "c1", "c2"):
                folder = WS + (f"view_{v}/" if v else "")
                required += prior_outputs(folder + f"law_outer{part}", folds=count)
        if not linked:
            outputs.append(WS + "law_independent_rank_bag_29_7.npz")
        add(tag, "white_independent_rank_folds", [*seeds, *(["--linked"] if linked else []), "--splits", count], outputs, required)
    for mode in ("pairs", "binary"):
        tag = "law_symmetric_" + mode
        add(tag, "white_pair_trees", [mode], [WS + tag + ".npz", *folds(tag + "_models", ("fold{k}.cbm",))], [*guard, *view_inputs])

    # Agreement heads use this additional view-law family. These views are
    # formed from the original pulse streams, rather than the raw-view path.
    crop_outputs = [WS + "lawcrop_c1_c2_rev.npz", *folds("lawcrop_c1_c2_rev_models", ("fold{k}.txt",))]
    crop_inputs = []
    for v in ("c1", "c2", "rev"):
        crop_outputs += tables("view_state_" + v)
        crop_outputs += arrays("pulse" + v + "_streams state" + v + "_streams")
        crop_outputs += tables("jump_law_pulse" + v + " jump_law_state" + v)
        if v != "rev":
            crop_inputs += [f"tune_out/var/clean_{p}_{v}.parquet" for p in SPLITS]
    add("lawcrop_c1_c2_rev", "white_crop_laws", ["c1", "c2", "rev", "--train"], crop_outputs, crop_inputs)

    neural = [
        ("lawshared", "white_shared_net", "lawshared_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawconditional", "white_mixture_program", "conditional_physics_models", ("fold{k}.pt", "fold{k}_input.npz")),
        ("lawpredictive", "white_predictive_density", "predictive_density_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawfourier", "white_fourier_kernel", "fourier_law_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawvectors", "white_conditional_weights", "conditional_weight_models", ("fold{k}.pt", "fold{k}_recipe.npz")),
        ("lawrankdensity", "white_rank_likelihood", "rank_likelihood_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawforecast", "white_forecast_density", "forecast_density_models", ("forecast{k}.pt", "head{k}.pt", "fold{k}.npz")),
        ("lawblock", "white_block_kernel", "block_kernel_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawretrieval", "white_retrieval", "lawretrieval_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawpatchforecast", "white_patch_forecast", "patch_forecast_models", ("fold{k}.pt", "fold{k}.npz")),
        ("lawrestart", "white_restart_density", "restart_density_models", ("density{k}.pt", "head{k}.pt", "fold{k}.npz")),
    ]
    for tag, module, folder, patterns in neural:
        extra = view_inputs if tag == "lawretrieval" else []
        if tag == "lawvectors":
            extra = folds("physical_ft_8")
        add(tag, module, [], [WS + tag + ".npz", *folds(folder, patterns)], extra)
    for seed in (7, 29):
        tag = f"law_rank_dependence_s{seed}"
        add(tag, "white_rank_dependence", [seed], [WS + tag + ".npz", *folds(tag + "_models")],
            prior_outputs(WS + f"law_outer{seed}"))
    for module, folder, names in (
        ("white_shock_quantiles", "shock_quantile_models", ("law_shock_quantile_control", "law_shock_quantiles")),
        ("white_ordered_paths", "ordered_path_models", ("law_path_control", "law_ordered_paths")),
    ):
        add(names[-1], module, [], [*[WS + n + ".npz" for n in names], *folds(folder)], prior_outputs(WS + "law_outer29"))
    tag = "law_weight_ensemble_s29"
    cache = WS + "frozen_local_rank_family_agreement/"
    add(tag, "white_weight_ensemble", [29], [WS + tag + ".npz", WS + tag + "_res.npz", *folds(tag + "_models")],
        [*view_inputs, *[p for v in ("", "c1", "c2") for p in prior_outputs(WS + (f"view_{v}/" if v else "") + "law_outer29")],
         cache + "full.npz", *[cache + f"s29_f{k}.npz" for k in range(5)]])
    return dict(status="Specialist recipes recovered; runtime validation and full head ordering pending", stages=stages)


if __name__ == "__main__":
    result = build()
    (ROOT / "pipeline/white_model_stages.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result['stages'])} retained specialist stages")
