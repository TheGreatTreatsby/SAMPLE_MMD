# 🚀 Domain Adaptation with Deep Feature Clustering for Pseudo-Label Denoising in Heterogeneous SAR Image Classification
This repository contains the implementation of the paper titled "Domain Adaptation with Deep Feature Clustering for Pseudo-Label Denoising in Heterogeneous SAR Image Classification". The proposed method addresses the domain shift problem in Synthetic Aperture Radar (SAR) image classification by combining domain adaptation with deep feature clustering for pseudo-label denoising. The approach achieves competitive performance on the SAMPLE dataset across three challenging evaluation scenarios.
<h2>🔍 Key Contributions</h2>
<ul>
  <li><span style="font-size: 18px;">Linear Kernel Maximum Mean Discrepancy (MMD)</span></li>
  <li><span style="font-size: 18px;">Deep Feature Clustering for Pseudo-Label Denoising</span></li>
  <li><span style="font-size: 18px;">End-to-End Pseudo-Label Self Training&Pure deep learning approach without handcrafted features</span></li>
</ul>

<h2>📂 Dataset</h2>
<p>The experiments are conducted on the SAMPLE dataset, which contains paired synthetic and real SAR images of 10 military vehicle targets. The dataset is divided into three evaluation scenarios:<p>
<p>Scenario I: Training on synthetic data (14-17°) and testing on real data (14-17°).<p>
<p>Scenario II: Training on synthetic data (14-16°) and testing on real data (17°).<p>
<p>Scenario III: Training on synthetic data (14-17°) and domain adapter with limited real data (14-16°) and testing on real data (17°).<p>


<h2>📊 Results</h2>
<p>The proposed method achieves the following results:</p>
<ul>
  <li><span style="font-size: 16px;">Scenario I: 98.05% accuracy</span></li>
  <li><span style="font-size: 16px;">Scenario II: 97.85% accuracy</span></li>
  <li><span style="font-size: 16px;">Scenario III: 98.65% accuracy</span></li>
</ul>
<p>The results reported here(also in paper) do not represent the best performance, but they are extensive. <p>
<p>In fact, due to the small number of SAMPLE data sets, even the average value of 20 results will fluctuate in a certain range. <p>
<p>We show some result files in the ./RESULT folder.<p>


<h2>🛠️ Code running</h2>
<p>Install the code base<p>
<p>Modify the path of the dataset<p>
<p>Run the Scenario I(or II,III).py directly.<p>
