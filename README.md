This project  trains CNN models to classify satellite imagery into 4 terrain types (agriculture, barrenland, grassland, urban) using SAR radar images from Sentinel-1, optical RGB images from Sentinel-2, and a late-fusion model that combines both. 

It classifying satellite imagery into 4 terrain types: agricultural land, barrenland, grassland, and urban. The dataset (Sentinel-1/2 paired images) has 16,000 image pairs, split 70/15/15 into train/val/test using a fixed seed for reproducibility.

The three models
SAR model takes a 1-channel (grayscale) radar image. Radar penetrates clouds and works at night — it captures surface texture and backscatter patterns. Your CNN has 5 conv blocks (Conv → BatchNorm → ReLU → MaxPool), each halving spatial resolution, taking 256×256 down to 8×8, then flattening to 8,192 features. A small head (Linear → ReLU → Dropout(0.5) → 4 logits) does the classification.
Optical model is identical in architecture but takes 3-channel RGB images. Optical captures colour and vegetation signatures that SAR can't — crops look green, urban is grey — but is blocked by cloud cover.
Fusion model is where Both branches are full CNNs producing 8,192-dimensional feature vectors. You concatenate them (16,384-dim) and pass through a shared fusion head. This is called late fusion because you merge at the feature level, not the pixel level(raw data). The key design choice is that you warm-started the fusion model's two branches from the single-modal checkpoints — so the CNNs already know how to interpret each modality before fusion training begins. Only the fusion head starts from scratch.
<img width="786" height="554" alt="image" src="https://github.com/user-attachments/assets/3866505e-41fd-4e86-bacf-f59bdeea2a14" />
