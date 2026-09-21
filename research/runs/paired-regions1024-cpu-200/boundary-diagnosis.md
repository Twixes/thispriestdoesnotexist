# Collar-adjacent pixel diagnosis

The region-balanced output matches brightness inside the manually traced tab but spills brightness into nearby clothing. On the same training identity at step200, the 15px and 30px exterior rings have MAE98.24 and99.47 out of255, compared with52.99 and35.44 for the broad-mask baseline. Mean luminance is150.61/133.43 versus target59.09/37.86. This is consistent with the native visual review of a broad neck patch rather than a crisp tab.

The5px ring remains bright in the target (mean116.65), so the raw tab trace is not an exact black/white segmentation boundary. Anti-aliasing, trace placement and target resizing matter. Do not interpret every exterior pixel as black garment. Three fixed radii (5,15,30px) were measured; none was selected by optimizing this output. Every ring is clipped to the original clothing mask and includes no protected face pixels.

`boundary-diagnosis.json` retains exact PNG/script/helper hashes and all values. The standalone `analyze_collar_boundary.py` uses saved images only, without loading a model or modifying training. This single-example diagnostic does not prove that a boundary loss will generalize. The active six-identity run retains its unchanged objective; its completed results will guide any later ablation.
