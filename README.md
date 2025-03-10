# Whales_Machine_Learning

This repository contains the time-frequency ridge data (in matresults) and functions (in matfun) to carry out the Bryde's whale call classification described in Tary et al. 2025.

Different functions are provided to perform (many need the data included in Matlab files in matresults):
  - the 4th order synchrosqueezing transform of signals (fsst_calculation.m, needs FSSTn toolbox)
  - t-SNE clustering (tSNE_example.m)
  - K-means clustering (kmeans_clustering.m)
  - SVM classification (svm_classification.m), with 50 tries of data order randomization and calculation of balanced accuracy
  - Statistical significance of SVM model (significance_rand_perm.m) using random permutation of data labels
  - Feature importance (feature_importance.m) using either permutation importance or SHAP values

The individual whale calls that were detected and located by Tary et al., 2024, can be found at https://doi.org/10.5281/zenodo.14998950.

For further details:
Hobbs, R., and Peirce, C. (2015). RRS James Cook JC114 Cruise Report. Online Report. https://www.bodc.ac.uk/resources/inventories/cruise_inventory/reports/jc114.pdf.

Tary, J. B., Peirce, C., Hobbs, R. W., Bonilla Walker, F, De La Hoz, C., Bird, A., and Vargas, C. A. (2024). “Application of a seismic network to baleen whale call detection and localization in the Panama basin – a Bryde’s whale example”, Journal of the Acoustical Society of America, 155(3), 2075-2086.

Tary, J. B., Peirce, C., and Hobbs, R. W. (2025). "Classification of Bryde’s whale individuals using high-resolution time-frequency transforms and support vector machines", Journal of the Acoustical Society of America.
