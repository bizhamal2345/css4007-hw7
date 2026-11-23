# HW7 submission — Blackjack

## AI-use declaration

I used OpenAI ChatGPT to help interpret the assignment, develop a worked
calculation guide, draft Python scripts, and check the fixed-history numerical
results. The agent implementations use NumPy/Python and Gymnasium game steps,
not a pre-built reinforcement learning algorithm. I reviewed the steps and
ran the commands myself. **I must still personally write, photograph and
commit the original paper calculations before the learning-code commit.**
Edit this declaration if it does not exactly reflect what you did.

## Setup and reproduction

- Python 3.10+ recommended.
- Install dependencies: `python -m pip install -r requirements.txt`.
- Fixed supplied histories: `python src/paper_replay.py`.
- Gymnasium experiment: `python src/sim_blackjack.py`.
- Generate this report: `python src/generate_submission.py`.
- Fixed policy is hit below 20, stick otherwise; discount=1.0 and TD/Q learning rate=0.1.
- Game environment: `Blackjack-v1`, `natural=False`, `sab=False`.

Installed Gymnasium 1.4.0; NumPy 2.4.6; Matplotlib 3.10.0.

## Part A — Count states and actions

Worked formulas and explanation: [paper/PART_A_B_WORKED.md](paper/PART_A_B_WORKED.md).

- Blackjack decision observation count: `10×10×2 = 200`; Q entries `200×2 = 400`.
- Durak unordered hand count: `C(36,6)=1,947,792`; candidate opponent hands `C(29,6)=475,020`.
- Tic-tac-toe board encodings: `3^9=19,683` (not the exact reachable count).
- Nim labeled pile encodings: `3×4×5=60` configurations, 120 with turn labels.
- Worked guide includes all legal action examples, hidden state discussion, and comparison table.

## Part B — Paper calculations

Worked calculation guide: [paper/PART_A_B_WORKED.md](paper/PART_A_B_WORKED.md).

**Original handwritten scans/photos must be added by the student and committed
before learning code.** Add their real relative paths here after photographing them.
Do not present the worked typed guide as an original handwritten scan.

## Part C — Reproduce the calculations

Program: [src/paper_replay.py](src/paper_replay.py).
Complete saved output: [outputs/paper-replay/paper_replay.txt](outputs/paper-replay/paper_replay.txt).
Full paper-vs-program table: [outputs/paper-replay/paper_vs_program.csv](outputs/paper-replay/paper_vs_program.csv).

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | --- | --- | --- | --- |
| Model-based | A | sweep 2 | -0.333333 | -0.333333 | 0.000000 | matches paper |
| Monte Carlo | A | one pass | -0.333333 | -0.333333 | 0.000000 | matches paper |
| TD(0) | A | final | -0.140030 | -0.140030 | -0.000000 | matches paper |
| Q-learning | A/hit | final | -0.139220 | -0.139220 | 0.000000 | matches paper |
Checked **62** reference values; **0** differences over numerical tolerance 1e-10. Floating point differences near 1e-16 are harmless.

## Part D — Simulated games

### Fixed-policy value estimates

Shared logs: `outputs/simulation/fixed_policy_logs/games_seed_*.csv`.
At each checkpoint (100, 1000, 10000 games) find state-value and
visit-count grids, separately for usable ace=0/1, in
`outputs/simulation/fixed_value_tables/seed_*/after_*_games/`.

Model-based estimates use empirical transition counts and synchronous policy
evaluation with tolerance 1e-6, maximum 10000 sweeps. Unobserved state-policy
pairs are **missing**, not observed zeros. Unknown successor values are marked
`missing_successor` and estimates depending on them are not reported as known.
MC uses every visit in completed episodes; TD(0) uses every observed transition.
Their values differ because MC uses full realized returns, TD bootstraps from
recent estimates, and model-based learning estimates a transition model first.
States visited rarely have noisy estimates, and more samples generally improve
coverage but need not improve every individual estimate monotonically.

### Q-learning and baseline evaluation

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | baseline | 1011 | 2962 | 6462 | 576 | 0 | -0.350000 |
| 11 | epsilon_0p2 | 1011 | 4204 | 4894 | 902 | 0 | -0.069000 |
| 11 | epsilon_0p0 | 1011 | 4119 | 4884 | 997 | 0 | -0.076500 |
| 22 | baseline | 1022 | 2935 | 6481 | 584 | 0 | -0.354600 |
| 22 | epsilon_0p2 | 1022 | 4204 | 4923 | 873 | 0 | -0.071900 |
| 22 | epsilon_0p0 | 1022 | 4242 | 4867 | 891 | 0 | -0.062500 |
| 33 | baseline | 1033 | 2950 | 6496 | 554 | 0 | -0.354600 |
| 33 | epsilon_0p2 | 1033 | 4131 | 4966 | 903 | 0 | -0.083500 |
| 33 | epsilon_0p0 | 1033 | 4155 | 4946 | 899 | 0 | -0.079100 |

### Plots and learned policies

- `outputs/simulation/training_mean_reward_epsilon_0p2.png` (training, epsilon 0.2).
- `outputs/simulation/training_mean_reward_epsilon_0p0.png` (training, epsilon 0).
- `outputs/simulation/evaluation_mean_reward.png` (frozen evaluation only).
- `outputs/simulation/q_learning/epsilon_*/seed_*/` contains both Q-action values,
  state-action visit counts, and greedy action/tie grids for ace=0 and ace=1.
- `outputs/simulation/run_parameters_and_versions.json` records versions and settings.

### Explain your results

The model-based method learns **transition probabilities and mean rewards**;
Monte Carlo and TD(0) learn **state values** for a supplied fixed policy;
Q-learning learns **action values** and can select an action from them.
A terminal step has no future value, so updates use only the immediate reward.
A usable ace makes certain totals safer to hit because an ace can change from
11 to 1 without busting. Epsilon 0.2 explicitly samples both actions;
epsilon 0 does not explore except that random ties may make different actions.
Unvisited Q-actions start at 0 but are not verified good actions.
With finite samples and random seeds, improvement is **not guaranteed**.

Comparison using all evaluation seeds (Q mean minus baseline mean):

| Training seed | Q policy | Baseline mean | Q mean | Difference | Improved? |
| --- | --- | --- | --- | --- | --- |
| 11 | epsilon_0p2 | -0.350000 | -0.069000 | 0.281000 | yes |
| 11 | epsilon_0p0 | -0.350000 | -0.076500 | 0.273500 | yes |
| 22 | epsilon_0p2 | -0.354600 | -0.071900 | 0.282700 | yes |
| 22 | epsilon_0p0 | -0.354600 | -0.062500 | 0.292100 | yes |
| 33 | epsilon_0p2 | -0.354600 | -0.083500 | 0.271100 | yes |
| 33 | epsilon_0p0 | -0.354600 | -0.079100 | 0.275500 | yes |
Q-learning improved on the baseline in **6 of 6** seed/policy comparisons.
