## Results: Convergence

Across all 10 models from 9 organizations, trait rankings are strongly correlated. Pairwise Spearman rank correlations range from ρ = 0.636 (Qwen3 vs. Trinity) to ρ = 0.906 (Claude Haiku 4.5 vs. GPT-5.1), with a median of ρ = 0.756 across all 45 model pairs (all p < 10⁻¹⁷). Spearman's ρ measures agreement in ordinal ranking — a natural fit for ELO-derived scores, where relative ordering is more meaningful than raw magnitude. A ρ of 0.76 indicates that two models largely agree on which traits to express more and which to suppress.

However, this overall correlation masks an important asymmetry in *where* models agree. To quantify this, we computed the standard deviation of each trait's rank across all 10 models, then grouped traits into tiers by average rank position. The results reveal a U-shaped pattern of convergence:

<!-- LaTeX source for Table 1 (ACL two-column format, booktabs) -->

```latex
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}lrcl@{}}
\toprule
\textbf{Rank Tier} & \textbf{$N$} & \textbf{Mean $\sigma$} & \textbf{Example Traits} \\
\midrule
Top 20         & 20 & \textbf{9.2}  & structured, systematic, precise \\
Ranks 21--50   & 30 & 18.5 & technical, elaborate, confident \\
Ranks 51--100  & 50 & 22.5 & reflective, decisive, verbose \\
Bottom 44      & 44 & 15.7 & excitable, passionate, competitive \\
\bottomrule
\end{tabular}
\caption{Mean rank standard deviation ($\sigma$) across 10 frontier models, grouped by trait tier. Lower $\sigma$ indicates stronger cross-model agreement on trait ranking. The top~20 most-expressed traits show 2.5$\times$ less variance than the middle tier (ranks 51--100).}
\label{tab:convergence}
\end{table}
```

Convergence is strongest at the top of the rankings. The 20 traits that models most prefer to express — structured, systematic, precise, methodical, concrete, analytical, disciplined — show a mean rank σ of just 9.2. For context, "structured" has an average rank of 2.6 across all models (σ = 2.6); every model places it in its top 9. These are not stylistic choices — they represent a shared conception of what a competent assistant *is*: organized, accurate, and methodical.

At the other extreme, models also converge on what to avoid. "Foolish" and "sycophantic" are the two most uniformly suppressed traits (rank σ = 3.1 and 3.0 respectively), with every model placing them in the bottom 15. This likely reflects a direct imprint of alignment training, which explicitly penalizes outputs perceived as unreliable, obsequious, or erratic. Yet even at the bottom, convergence (σ = 15.7) is markedly weaker than at the top (σ = 9.2), suggesting that alignment training applies a stronger positive signal ("be structured, be precise") than a negative one ("don't be foolish"), or at least that models interpret the negative signal less uniformly.

The middle tier — ranks 51 through 100 — is where models diverge most (σ = 22.5). These are predominantly stylistic and dispositional traits: poetic (σ = 36.8), contemplative (σ = 34.9), simplistic (σ = 34.4), playful (σ = 31.1). Whether to be philosophical or pragmatic, verbose or concise, conservative or progressive — these are the axes along which frontier models develop distinct personalities. Notably, this is also where the principal components of variation concentrate (see PCA analysis, Section X), confirming that inter-model personality differences are stylistic rather than competence-related.

The U-shaped convergence pattern carries an implication for the field: the "helpful assistant" archetype is not merely a set of guardrails against bad behavior. It is an affirmative personality profile — systematic, precise, disciplined — that labs have independently converged upon despite differing architectures, training corpora, and organizational cultures. Five of the ten models tested (Kimi K2, Trinity, Grok 4, DeepSeek V3.2, and Claude Haiku 4.5) cluster so tightly in trait space that their personality profiles are nearly interchangeable. The remaining variation is confined to stylistic dimensions that do not threaten helpfulness or safety.
