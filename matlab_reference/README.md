# MATLAB Reference Code

This folder contains the original MATLAB source code exactly as transcribed in the appendices of the [thesis](https://ri-ng.uaq.mx/handle/123456789/10410), kept here for traceability so that anyone comparing the PyTorch reimplementation against the original can do so directly.

- `apendice1_simple_mlp.m` — Simple multilayer perceptron (patternnet) for LG mode reconstruction.
- `apendice2_downsampling.m` — Block-averaging downsampling helper function used by Appendix 1.
- `apendice3_simple_cnn.m` — Custom 3-layer convolutional neural network.
- `apendice4_efficientnetb0.m` — Transfer learning with EfficientNet-B0.
- `apendice5_resnet.m` — Transfer learning with ResNet18/50/101.
- `apendice6_gui.m` — MATLAB App Designer GUI (`GUILaguerreGaussBeams`).

These files are not executable as part of this Python repository; they are included purely as a reference for the reimplementation in `src/` and `gui/`.
