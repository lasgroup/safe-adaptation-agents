# safe-adaptation-agents
Baseline algorithms for the `safe-adaptation-gym` benchmark.

## Install
1. Open a new terminal and `git clone` the repo.
2. Create a new environment with your favorite environment manager (venv, conda). Make sure to have it set up with `python >= 3.8.13`.
3. Install dependencies with `cd safe-adaptation-agents && pip install .`.

## Run
Let's reporduce the experiments for the benchmark.
The following command runs the on-policy algorithms:
```bash
 python scripts/adaptation_experiment.py --configs defaults multitask on_policy"  --agent <insert agent> --seed <insert seed>
```
where $\texttt{seed} \in \lbrace1,2, \dots, 10\rbrace$ and $\texttt{agent} \in \lbrace\$`maml_ppo_lagrangian, rl2_cpo, rarl_cpo`$\rbrace$
Similarly, to run the model-based algorithms:
```bash
 python scripts/adaptation_experiment.py --configs defaults multitask model_based"  --agent <insert agent> --seed <insert seed>
```
where $\texttt{agent} \in \lbrace\$`la_mbda, carl`$\rbrace$
To run an agent on a specific task:
```bash
 python scripts/no_adaptation_experiment.py --configs defaults no_adaptation" --agent <insert agent> --task <insert task>
```
where $\texttt{task} \in \lbrace$`go_to_goal, dribble_ball, collect, push_box, haul_box, press_buttons, catch_goal, roll_rod`$\rbrace$
In our experiments, we used `cpo` as our agent.

More generally, every parameter in the `configs.yaml` file can be easily changed when running either `scripts/no_adaptation_experiment.py` or `scripts/adaptation_experiment.py` by appending `--parameter` when running the script.





