Focus on developing a deep learning-based system for accurate identification and analysis of human actions in varied environments.

It is aimed at applications in surveillance, security, sports, and fitness.

Utilizes advanced models like AlexNet3D and VGG3D for processing video data.

Employs SelfiSegmentation from the cv-zone library for background and human masking in videos.

Trained on a dataset of videos representing various human actions.

Experimentation was conducted on the UCF101 dataset, with adaptations for computational limitations.

Performance analysis of models in different scenarios, considering accuracy, precision, and recall.

Discuss the relative merits and limitations of simpler CNN models versus advanced Transformer-based models.

Emphasizes the need for continuous adaptation in deep learning and video analysis.

Here are the summarized bullet points:

1. **AlexNet Performance Analysis:**
   - Without Mask: Accuracy of 0.36 and Precision of 0.3751.
   - With Background Mask: Accuracy of 0.31 and Precision of 0.2061.
   - With Human Mask: Accuracy of 0.25 and Precision of 0.2819.
   - Recall values: 0.3751 (Without Mask), 0.2061 (With Background Mask), 0.2819 (With Human Mask).

2. **VGG Model Performance:**
   - Without Mask: Accuracy of 0.39 and Precision of 0.3526.
   - With Background Mask: Accuracy of 0.17 and Precision of 0.2385.
   - With Human Mask: Accuracy of 0.20 and Precision of 0.2385.
   - Recall values: 0.3526 (Without Mask), 0.2385 (With Background and Human Masks).

3. **Comparative Insights:**
   - AlexNet and VGG models show varied performance under different conditions (with/without masks).
   - Overall, higher accuracy and precision are observed without masks for both models.
   - Recall values suggest a consistent trend in the ability of models to correctly identify relevant instances.
