---
title: Tuning properties of M1 cells during classical conditioning
author: Emerson Harkin
date: November 18, 2020
geometry: margin=1in
...

# Cell-level analysis

To better understand how the neural representation of reward in M1 changes over
the course of learning, we began by comparing the structure of the
trial-related calcium fluorescence on training days 1 and 7. By applying
principal components analysis to the Z-scored and trial-averaged fluorescence
of all 1925 cells recorded on day 7 and regressing the fluorescence of each
cell onto the top 50 components (which collectively captured 93.0% of the
trial-averaged fluorescence on day 7), we obtained a compressed signature of
the calcium fluorescence of each cell in terms of a 50-dimensional vector of
component weights $w$. To assess the extent to which the calcium fluorescence
of each cell reflected each stage of the classical conditioning experiment, we
compared the weight vector of each cell to a set of weight vectors representing
the tone and reward-delivery trial stages using the cosine of the angle between
the two vectors as a measure of how closely the compressed fluorescence
signature of each cell reflected a particular part of the trial (see fig. 1).
This angular distance metric, which we refer to as the tuning coefficient,
ranges from -1 to 1 and can be interpreted similarly to a correlation
coefficient; 1 indicates that a cell perfectly reflects a given trial stage by
increasing its fluorescence, -1 indicates that the cell decreases its
fluorescence, and 0 indicates no linear relationship between fluorescence and
the trial stage in question (see fig. 2). On day 7, the cells in our dataset
exhibited a wide range of tone and reward tuning coefficients, with a small
proportion of cells being strongly positively or negatively tuned to tone or
reward and a larger proportion exhibiting weaker tuning (tuning coefficients
near -1 or 1, see figs. 3-4). The overall distribution of tone tuning
coefficients was stable from day 1 to day 7 (KS test p=0.238, N=1926 cells on
day 1 and N=1925 cells on day 7), indicating that the same proportions of cells
were strongly or weakly activated or inhibited by tone before and after
conditioning. In contrast, there was an overall shift towards a greater degree
of activation in response to reward (two-sided KS test p&lt;0.05 followed by
one-sided KS test p=8.15e-14, N=1926 cells on day 1 and N=1925 cells on day 7).
Together, these observations show that neurons in M1 exhibit trial-averaged
activity that is associated with tone and reward delivery in our classical
conditioning task, and that the aggregate response to reward but not tone
changes over the course of learning.

![Quantifying tuning using an angular distance metric. The reward tuning
coefficient is the cosine of the angle between an indicator function for the
time of reward delivery (shown as a square step in yellow) and the Z-scored
fluorescence of a particular cell (purple) within principal component space.
The tuning coefficient is close to 1 when the angle is small, close to -1 when
the angle is close to 180 deg, and close to 0 when the fluorescence is
orthogonal to the timing of reward
delivery.](../img/graphics/angular_similarity.png)

![Sample tuning coefficient traces. (Cosine similarity is another name for the
tuning coefficient.)](../img/graphics/sample_responsive_cells.svg)

![Tone tuning heatmap. Trial-averaged fluorescence of 1925 cells recorded on
day 7 ordered according to the value of the tone tuning
coefficient.](../img/graphics/fluo_ordered_by_tone.svg)

![Reward tuning heatmap. Trial-averaged fluorescence of 1925 cells recorded on
day 7 ordered according to the value of the reward tuning
coefficient.](../img/graphics/fluo_ordered_by_reward.svg)

The apparent stability of the statistics of tone tuning led us to wonder
whether this population-level stability reflects an underlying stability in the
tuning coefficients of individual neurons, or whether specific populations of
neurons might exhibit opposing changes in tuning that cancel out at the level
of the population. When we compared the trial-resolved tuning coefficients of
each cell on days 1 and 7, we found that 61.5% (N=1926 neurons) did not exhibit
a significant change in the tone tuning coefficient. In contrast, only 45.7% of
neurons (N=1926) had a stable reward tuning coefficient from day 1 to day 7.
This suggests that the relative population-level stability in the degree of
tone tuning is partly due to the stability of tuning at the level of individual
cells.

Of the neurons that did significantly change their tone tuning coefficients
from day 1 to 7, less than half of pyramidal neurons exhibited a negative
change in the tone tuning coefficient (42.2% of N=422 pyramidal neurons with a
significant change in the tone tuning coefficient) while interneuron cell types
overwhelmingly decreased their tone tuning coefficients (95.4% of N=174 PV
neurons and 82.1% of N=145 VIP neurons; see Table 1). In contrast, the changes
in reward tuning coefficients of both pyramidal neurons and PV interneurons
were approximately evenly split between increases and decreases (57.2% of N=659
pyramidal and 53.6% of N=153 PV cells with significant changes in reward tuning
coefficients exhibited a decrease) whereas the reward tuning coefficients of
VIP neurons mostly increased (27.4% of N=234 VIP neurons exhibited a decrease;
see Table 2). Overall, these trends suggest that among neurons with changing
task-related activity, pyramidal neurons exhibit heterogenous changes in tone
and reward tuning over the course of learning, while PV and VIP neurons become
less activated and/or more inhibited by tone, and VIP neurons become more
activated and/or less inhibited by reward.


-------------------------------------------------------------------------------

Table 1. Number of cells with a significant change in the tone tuning
coefficient from day 1 to day 7. A significant change was defined as a
one-sided Mann-Whitney U-test p&le;0.05 on the trial-resolved tone tuning
coefficients for a particular cell (N=AROUND THIRTY trials per cell per day).
Accordingly, tuning increase or decrease rates of approximately 5% simply
reflect the false positive rate of the U-test and should be ignored; only
increase or decrease rates of &gt;5% are meaningful. For example, this means
that the low tone tuning increase rates in PV and VIP cells should be
interpreted as indicating that *none* of these cells have an increasing tone
coefficient, and any apparent increases are due only to noise. The values in
the percent tuning decrease column are calculated as the number of tone tuning
decrease cells (col 3) divided by the number of cells with either an increase
or decrease in tuning coefficient (col 3 + col 4). Where indicated, the
uncertainty represents an approximate 95% confidence interval on the
proportion, calculated as two standard errors ($2 \sqrt{\frac{p(1 - p)}{N}}$).

| Cell type | Num. cells | Tone tuning decrease | Tone tuning increase | Pct. tuning decrease |
| :-------: | :--------: | :------------------: | :------------------: | :------------------: |
| PYR       | 1207       | 178 (14.7% +/- 2.0%) | 244 (20.2%)          | 42.2% +/- 4.8%       |
| PV        | 312        | 166 (53.2%)          | 8 (2.6%)             | 95.4% +/- 3.2%       |
| VIP       | 407        | 119 (29.2%)          | 26 (6.4%)            | 82.1%                |

Table 2. Number of cells with a significant change in reward tuning coefficient
from day 1 to day 7. See caption of Table 1 for details.

| Cell type | Num. cells | Rew. tuning decrease | Rew. tuning increase | Pct. tuning decrease |
| :-------: | :--------: | :------------------: | :------------------: | :------------------: |
| PYR       | 1207       | 377 (31.2%)          | 282 (23.4%)          | 57.2% +/- 3.8%       |
| PV        | 312        | 82 (26.3%)           | 71 (22.8%)           | 53.6% +/- 8.0%       |
| VIP       | 407        | 64 (15.7% +/- 3.6%)  | 170 (41.8%)          | 27.4% +/- 5.8%       |

-------------------------------------------------------------------------------

# Mouse-level analysis

In the previous section, we identified several trends in the dynamics of the
calcium responses of M1 cells during learning using a technique designed to
capture the tone- and reward-associated activity of multiple cell types in an
unsupervised manner. Our analysis also highlighted the diversity of responses
to conditioned and unconditioned stimuli both within and across cell types.
Because this analysis was based on data pooled across multiple mice, an
important caveat associated with these trends is that some of the apparent
response diversity within cell types could be due to mouse-to-mouse differences
(eg from experimental factors such as level of GCaMP expression, imaging site,
as well as behaviour). Indeed, when we examined the distribution of tuning
coefficients on day 7 stratified by mouse in addition to trial component (tone
vs reward) and cell type, we observed clear diversity in the tuning of
individual cells within each mouse as well as a significant level of
mouse-to-mouse variability (see fig. 5). Inspecting the distribution of changes
in the tuning coefficients from day 1 to day 7 stratified by the same set of
factors revealed trends that were broadly consistent with our pooled analysis,
but also highlighted apparent differences between mice (for example, the tone
tuning coefficients of pyramidal neurons shifted significantly towards -1 in
4/6 mice (Wilcoxon signed-rank test p&lt;0.05 in each case), but shifted
towards 1 in one mouse).

We identified changes in the tuning of each cell type that were consistent
across mice by CANDICE'S ANALYSIS GOES HERE

![Tuning coefficients on day 7 stratified by mouse, trial component, and cell
type.  Curves represent Gaussian kernel density estimates of the distribution
of tuning coefficients. Each curve is one mouse. Dots above the curves indicate
the median for the corresponding mouse; dots are randomly jittered along the
y-axis for clarity. Values close to 1 indicate excitation associated with tone
or reward, values close to -1 indicate inhibition, and values close to 0
indicate no association.](../img/graphics/per_mouse_similarity.svg)

![Change in the tuning coefficient from day 1 to day 7 stratified by mouse,
trial component, and cell type. Values &gt;0 indicate an increase in excitation
or decrease in inhibition, values &lt;0 indicate the reverse, and values close
to 0 indicate no change.](../img/graphics/per_mouse_similarity_change.svg)
