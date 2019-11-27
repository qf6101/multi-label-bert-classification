### Multi-label Bert Classification

This implementation adds useful features on bert classification: 

1. Multi-label
2. Focal loss weighting
3. Auto cross-label data synthesis
4. Adding exclude loss part among specific labels
5. Upsampling
6. Robust mean over all positive or negative loss
7. Generating very fast inference-time model

N.B. I deleted the efficient model service part (about 500 qps per 1080ti gpu).
