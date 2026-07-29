# Project Report — Deer vs Antelope CNN Classifier

**Dataset source:** Real photographs were sourced from the public Kaggle
dataset "Animal Image Dataset — 90 Different Animals" (iamsouravbanerjee),
filtered to the native `deer` and `antelope` classes — no synthetic images
were used.

**How to use the application:** Open the deployed Streamlit URL, upload a
`.jpg`/`.png` photo of a deer or an antelope, and the app returns the
predicted class with a confidence score within seconds.

**Challenges encountered:** Class images varied in resolution, background
clutter, and pose, which initially caused overfitting; this was mitigated
with data augmentation and dropout. Some deer/antelope species are visually
similar (e.g., similar coat colour and antler shape), causing occasional
misclassifications, especially in side-profile or low-light images.

**Possible improvements:** Expanding the dataset with more species-diverse
images, applying stronger regularization or a larger backbone (EfficientNet),
and adding Grad-CAM visual explanations to increase user trust in predictions.
