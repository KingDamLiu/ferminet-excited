# from ferminet1 import base_config
# from ferminet1.utils import system
# from ferminet1 import train


# # cfg = base_config.default()
# # cfg.system.electrons = (15,11)  # (alpha electrons, beta electrons)
# # cfg.system.molecule = [system.Atom('Fe', (0, 0, 0))]


# # O3
# cfg = base_config.default()
# cfg.system.electrons = (12,12)  # (alpha electrons, beta electrons)
# cfg.system.molecule = [system.Atom('O', (0, 0, 0)),system.Atom('O', (0, 0, 1.26881)),system.Atom('O', (1.12933, 0, -0.57836))]

# # # O
# # cfg = base_config.default()
# # cfg.system.electrons = (12,12)  # (alpha electrons, beta electrons)
# # cfg.system.molecule = [system.Atom('O', (0, 0, 0)), system.Atom('O', (0, 0, 1.20780)), system.Atom('O', (20, 0, -0.57836))]

# train.train(cfg)

import sys

from absl import logging
from ferminet.utils import system
from ferminet import base_config
from ferminet import train

# Optional, for also printing training progress to STDOUT.
# If running a script, you can also just use the --alsologtostderr flag.
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# # Define H2 molecules
# cfg = base_config.default()
# cfg.system.electrons = (15,9)  # (alpha electrons, beta electrons)
# cfg.system.molecule = [system.Atom('O', (0, 0, 0)),system.Atom('O', (0, 0, 1.26881)),system.Atom('O', (1.12933, 0, -0.57836))]

# O2
# cfg = base_config.default()
# cfg.system.electrons = (10,6)  # (alpha electrons, beta electrons)
# cfg.system.molecule = [system.Atom('O', (0, 0, 0)), system.Atom('O', (0, 0, 1.20780))]

# Fe
cfg = base_config.default()
cfg.system.electrons = (15,11)  # (alpha electrons, beta electrons)
cfg.system.molecule = [system.Atom('Fe', (0, 0, 0))]

# Set training parameters
cfg.batch_size = 2048
cfg.pretrain.iterations = 0
cfg.network.determinants = 32
cfg.log.save_path = 'logs/Fe'

train.train(cfg)