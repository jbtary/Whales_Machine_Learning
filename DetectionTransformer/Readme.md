# Detection of Bryde's whale calls using retrained EQTransformer

In this folder are included the main codes to perform the ML procedure developed by S. F. Poveda and described in Tary et al., 2025 (see below).

In the jupyter notebooks are included the construction of the training dataset using the original training examples and data augmentations (01_traininig_data.ipynb, 02_to_seisbench_format.ipynb), the training of the EQTransformer model (Mousavi et al., 2020) and its modifications (03_training.ipynb), and how the evaluation dataset was constructed (05_evaluation_dataset.ipynb).

The ML procedure is based on the Seisbench architecture (Woollam et al., 2022).

Also included are:
  - The model weights after training (ep*.pt files, final weights used for detection are those in ep105_wc_weights.pt)
  - The matlab file used to calculate evaluation metrics (main_eval.m), using the evaluation dataset and its characteristic functions (few examples are in the EvalMetrics folder). Final results are included in the file eval_metrics_15112024_2.mat.

Please cite these references in the case of re-using some of these codes:

Tary, J. B., Poveda, S. F., Li, K. L., Peirce, C., Hobbs, Richard W., and Vargas, C. A. (2025). "Detection and localization of Bryde’s whale calls using machine learning and probabilistic back-projection," The Journal of the Acoustical Society of America, 158(2), 1386-1397.

Woollam, J., Munchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Bornstein, T., and Soto, H. (2022). "SeisBench: A toolbox for machine learning in seismology," Seismol. Res. Lett. 93(3), 1695–1709.

Mousavi, S. M., Ellsworth, W. L., Zhu, W., Chuang, L. Y., and Beroza, G. C. (2020). "Earthquake transformer: An attentive deep-learning model for simultaneous earthquake detection and phase picking," Nat. Commun. 11(1), 1–12.
