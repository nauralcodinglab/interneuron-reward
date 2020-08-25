import numpy as np


class TrialBasisFunctions:
    def __init__(
        self,
        num_frames,
        baseline_duration=2.0,
        tone_duration=1.0,
        delay_duration=1.5,
        reward_duration=0.25,
        frame_rate=30.0,
    ):
        self.baseline_duration = baseline_duration
        self.tone_duration = tone_duration
        self.delay_duration = delay_duration
        self.reward_duration = reward_duration

        self.time = np.arange(0.0, num_frames) / frame_rate
        self.baseline = self.time < self.baseline_duration
        self.tone = (self.time >= self.baseline_duration) & (
            self.time < (self.baseline_duration + self.tone_duration)
        )
        self.delay = (
            self.time >= (self.baseline_duration + self.tone_duration)
        ) & (
            self.time
            < (
                self.baseline_duration
                + self.tone_duration
                + self.delay_duration
            )
        )
        self.reward = (
            self.time
            >= (
                self.baseline_duration
                + self.tone_duration
                + self.delay_duration
            )
        ) & (
            self.time
            < (
                self.baseline_duration
                + self.tone_duration
                + self.delay_duration
                + self.reward_duration
            )
        )

    @property
    def time_from_tone_start(self):
        self.time - self.baseline_duration

    @property
    def time_from_reward_start(self):
        self.time - (
            self.baseline_duration + self.tone_duration + self.delay_duration
        )


def ols(X, y, intercept=True):
    """Ordinary least-squares regression."""
    X = np.asarray(X)
    y = np.asarray(y)

    assert X.ndim <= 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]

    XTX = np.dot(X.T, X)

    if intercept:
        # Implicitly fit an intercept by centering y.
        y = y * 1.0 - y.mean()

    XTy = np.dot(X.T, y)
    coeffs = np.linalg.solve(XTX, XTy)

    return coeffs


def cos_similarity(a, b):
    """Cosine of the angle between two vectors."""
    assert np.ndim(a) == 1
    assert np.ndim(b) == 1
    assert len(a) == len(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
