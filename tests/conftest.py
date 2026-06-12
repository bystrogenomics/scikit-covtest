import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None
