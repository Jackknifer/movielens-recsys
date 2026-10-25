import sys
from pathlib import Path

import numpy as np

from ranking.ranking_predict import load_ranking_artifacts
from training.ranking_dataset import load_ranking_dataset


dataset = load_ranking_dataset(Path(sys.argv[1]))
ranking = load_ranking_artifacts(Path(sys.argv[2]))
inputs = {name: values[:32] for name, values in dataset.inputs("validation").items()}
np.save(sys.argv[3], ranking.model(inputs, training=False).numpy())
