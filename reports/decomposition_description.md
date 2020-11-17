As the animals progressed in the classical conditioning experiment, we
hypothesized that some of the cell types in motor cortex would play changing
roles in representing task structure. To test this hypothesis, we characterized
the representation of task structure at the end of the experiment and asked
whether the representation at the beginning of the experiment was similar. We
began by constructing a low-dimensional "fingerprint" of the activity of each
neuron at the end of the experiment using an unsupervised dimensionality
reduction method (see details below). Next, we calculated the corresponding
fingerprints for a set of idealized neurons whose activity perfectly reflected a
particular component of the behavioural task. By comparing the fingerprints of
the real and idealized neurons using a measure of angular similarity (1 for
perfect task encoding, -1 for encoding a mirror image, and 0 for no relationship
with task structure), we were able to quantify the extent to which the
activity of each neuron reflected each part of the task.

To
characterize the task representation at the end of the experiment, we began by
expressing the trial-averaged calcium fluorescence of each cell as a weighted
sum of a set of $D$ fluorescence primitives $$ F_i(t) = \sum_{j=1}^D w_{ij}
v_j(t), $$ where $F_i(t)$ is the trial-averaged and Z-scored calcium
fluorescence of cell $i$ over time, $v_j(t)$ is a fluorescence primitive, and
$w_{ij}$ is a weight that sets the contribution of primitive $j$ to the overall
fluorescence of cell $i$. In this framework, the fluorescence primitives
$v_j(t)$ represent patterns of activity shared across many neurons, and the
$w_{ij}$ associated with each cell represent a low-dimensional "fingerprint" of
its individual activity. Using an unsupervised method (PCA), we obtained a set
of 50 such primitives and corresponding weights which collectively accounted for
PERCENTAGE of the variance in the trial-averaged fluorescence across the NUMBER
OF CELLS included in our analysis. Next, we calculated the weight "fingerprints"
associated with each component of our behavioural task and compared them with
the fingerprints of each neuron using the cosine of the angle between the sets
of weights as a measure of similarity. This cosine similarity measure ranges
from -1 to 1, where 1 implies that the cell encodes a particular component of
the task very strongly, -1 implies that the cell encodes a mirror image of that
task component, and 0 implies that the activity of the cell has no relationship
with the task component at hand.

decomposing the trial-averaged calcium fluorescence
of all >4000 cells into a set of activity primitives or principal components
using PCA. The top

assumed
that the neural representation of the task in motor cortex would converge to
something.

