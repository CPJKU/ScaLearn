# src/scripts

This folder contains all the scripts needed to replicate our results.

When changing to (XLM-)RoBERTa-large, just change the model string from "base" to "large". Note that, in that case, we run with fewer seeds. This, then, must also be changed in the scripts.

When training a task transfer layer (ScaLearn and its variations + AdapterFusion), you need to first train single-task adapters (``st-a``), and change the locations in the ``config.json`` files and ``--output_dir`` accordingly (also for few-shot learning).
The same goes for probing analyses.

The 4 variations of ScaLearn are named and can be used in the scripts as follows:
- **ScaLearn**: ``scalearn_2_avg_d03``
- **ScaLearnUniform**: ``scalearnUniform_2_avg_d03``
- **ScaLearn++**: ``scalearnPP_2_avg_d03``
- **ScaLearnUniform++**: ``scalearnUniformPP_2_avg_d00``

To modify these configurations, you can edit the ``configuration.py`` in ``adapter-transformers/src/transformers/adapters``.