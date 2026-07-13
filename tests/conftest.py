import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

plt.show = lambda *args, **kwargs: None
