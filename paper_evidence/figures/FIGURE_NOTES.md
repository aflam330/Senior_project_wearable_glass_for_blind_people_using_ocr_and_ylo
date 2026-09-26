# Figure notes

Axes are labeled. Captions are not drawn inside the figures.
Figure 1 draws the measured authenticity path in blue. Grey boxes are other programs in this repository. They were not scored on the JaalTaka authenticity test. Microphone and tactile sensing are not in the repository and are not drawn.
Figure 4 and 5 place each view count at the prefix model's measured host median latency for that count, from results/qduig/edge/edge.json. That file is this Windows CUDA host, not a Raspberry Pi. The same latency is used for every model because separate timings were not in that file. Latency already grows with the view count, so the axis is that measurement rather than view count times a constant.
Figure 6 is written by the calibration script when reliability bins exist.
Figure 7 is the collapsed acquisition policy from eval_cost_policy.py.
Figure 8 uses results/robustness/severity_seed42.json. Each panel's x-axis is the parameter passed to apply_corruption, three values per corruption, at 6 views. For low light, JPEG quality, and scale, a larger parameter is not a stronger corruption. Gaussian blur at severity 3 and low light at 0.35 match the earlier single-severity file.
Figure 10 shows test notes the seed-42 PRMVT checkpoint got wrong at 1 view.
