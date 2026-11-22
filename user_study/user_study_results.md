# User study results (PIE-Bench change background, 16 pairs, 6 raters, 96 judgments per criterion)

| Criterion | Ours | MasaCtrl | Tie | p (sign test, ties excluded) |
|---|---|---|---|---|
| Image quality | 45.8% | 30.2% | 24.0% | 0.10 |
| Prompt fidelity | 72.9% | 17.7% | 9.4% | < 0.001 |
| Structure preservation | 22.9% | 54.2% | 22.9% | < 0.001 |

Per-question majority (16 questions):
- Prompt fidelity: ours wins 12, MasaCtrl wins 2, tie 2
- Structure preservation: ours wins 4, MasaCtrl wins 9, tie 3
- Every one of the 6 raters preferred ours more often than MasaCtrl on prompt fidelity.

Caveat: p-values treat the 96 judgments as independent; they come from 6 raters x 16 pairs, so read them as indicative.

## LaTeX table

\begin{table}[t]
\centering
\small
\begin{tabular}{lccc}
\toprule
Criterion & Ours & MasaCtrl & Tie \\
\midrule
Image quality & \textbf{45.8\%} & 30.2\% & 24.0\% \\
Prompt fidelity & \textbf{72.9\%} & 17.7\% & 9.4\% \\
Structure preservation & 22.9\% & \textbf{54.2\%} & 22.9\% \\
\bottomrule
\end{tabular}
\caption{User study on 16 randomly sampled PIE-Bench \emph{change background} pairs (6 participants, 96 judgments per criterion). Percentages are the share of pairwise judgments preferring each method.}
\label{tab:user_study}
\end{table}

## Paragraph for the report

\paragraph{User study.} We complement the automatic metrics with a blinded pairwise user study. We sampled 16 image pairs from the PIE-Bench \emph{change background} category (two from each of its eight sub-groups) using a fixed random seed. For each pair, participants saw the source image, the target prompt (with PIE-Bench's editing instruction shown only to make the intended edit easier to identify), and the outputs of MasaCtrl and our method in random left/right order, balanced so that each method appeared as ``A'' in half of the questions. They chose the better result, or ``tie'', for image quality, prompt fidelity, and preservation of the source subject. Six participants gave 96 judgments per criterion (Table~\ref{tab:user_study}). Our method was preferred for prompt fidelity in 72.9\% of judgments against 17.7\% for MasaCtrl (sign test, $p<0.001$), winning the majority vote on 12 of the 16 pairs, and every participant favored it on this criterion. Image quality showed a smaller, non-significant preference for our method (45.8\% vs.\ 30.2\%, $p=0.10$). In contrast, MasaCtrl was preferred for structure preservation (54.2\% vs.\ 22.9\%, $p<0.001$). This reflects the intended trade-off: releasing the background from the source image lets it follow the new prompt, but it also weakens the constraint that keeps the edited subject identical to the source. Because the study is small and the judgments are not independent, we treat these results as supporting evidence for the automatic metrics rather than a standalone result.
