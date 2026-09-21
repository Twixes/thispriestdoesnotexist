# Fresh-preservation band versus saved paired masks

At1024px, the current top75% fresh-preservation band is rows0–767 inclusive,786,432 pixels. **No raw white-tab trace or effective tab overlaps it** in either the current six training pairs or the proposed nine. Clothing overlap exists, principally for b2-020 and b2-055. This is geometric support evidence only: fresh preservation samples a different random latent, so an overlapping coordinate is not direct same-image contradictory supervision or proof of a training failure cause.

The exact current manifest is `research/data/paired7/manifest-proposed-v2.json`, SHA256 `9c27b19b0b7cfea4ca797a488b5d249410f3988b512373f713a599f02164bab6`. Its hash matches the recorded config of `research/runs/paired-regions1024-sixpair-600`. The exact proposed manifest is `research/data/paired10/manifest-proposed.json`, SHA256 `8d0660f1ccf5115a356e9e6b479e4d225c1e9993402fc71575b0fcb9ed8dcff1`. Held-out028 is excluded from both calculations. No active files were edited.

| Training pair | Clothing pixels | Clothing in band | Fraction of clothing in band | Effective tab pixels | Tab in band | Clothing objective spatial mass in band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|calibration-original|141,212|4,217|2.986%|7,198|0|1.573%|
|000|121,997|1,539|1.262%|4,248|0|0.654%|
|030|64,975|0|0%|4,204|0|0%|
|b2-055|267,793|56,894|21.246%|6,248|0|10.877%|
|b2-051|90,269|13|0.014%|4,924|0|0.008%|
|b2-020|314,638|118,600|37.694%|1,790|0|18.955%|
|b2-042 (proposed)|44,928|0|0%|2,016|0|0%|
|b2-058 (proposed)|130,594|7,108|5.443%|4,787|0|2.825%|
|b2-059 (proposed)|81,328|0|0%|4,262|0|0%|

The clothing objective is `0.5*mean_absolute_error(tab) + 0.5*mean_absolute_error(rest_clothing)`. Therefore the last column is `0.5*area(tab∩band)/area(tab) + 0.5*area(rest∩band)/area(rest)`. It measures the fraction of normalized spatial coefficient mass in that band under a unit-error convention, **not observed loss, gradient magnitude, or optimization conflict**. Equal region weighting means raw pooled pixel counts alone do not represent the objective.

| Cohort and region | Sum of region pixels across seeds | Sum in band | Pooled area fraction in band | Mean per-seed region fraction in band |
| --- | ---: | ---: | ---: | ---: |
|Current6: clothing|1,000,884|181,263|18.110%|10.534%|
|Current6: effective tab|28,612|0|0%|0%|
|Current6: remaining clothing|972,272|181,263|18.643%|10.689%|
|Proposed9: clothing|1,257,734|188,371|14.977%|7.627%|
|Proposed9: effective tab|39,677|0|0%|0%|
|Proposed9: remaining clothing|1,218,057|188,371|15.465%|7.754%|

The trainer samples paired indices uniformly. Its expected per-pair normalized clothing mass in this band is consequently5.344% for current6 and3.877% for proposed9, entirely from the remaining-clothing half of the objective. The corresponding mean protected-source component mass in the band is86.227% and84.658%; this merely describes its support, since the protected complement is normalized separately. The recorded weights are clothing1, protected1 and fresh1. Each such component uses its own normalization and examples; their coordinate masses are not interchangeable gradient norms.

A unit-normalized fresh-band loss allocates3.841% of its spatial mass to the average current6 clothing footprint and2.661% to the average proposed9 footprint. These are saved-pair coordinate overlays, **not measurements of clothing in the actual fresh images**. The union of all clothing overlaps is132,522 band pixels (16.851% of the band) in both cohorts;058 adds overlap already covered by previous training masks. That union is a coverage bound, not a sampling-weighted statistic.

This audit provides no evidence that the fresh band directly supervises the observed target tabs back to original clothing: all traced tabs sit below it. Shared generator parameters could still couple changes across coordinates or unseen identities; evaluating that requires a controlled experiment, not inference from overlap counts. It also cannot determine whether a random unseen face has high clothing or facial anatomy within the top75% band. No recipe change is justified solely by these counts.

Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 research/.venv/bin/python research/reviews/fresh-band-overlap/measure.py`. The script uses NumPy/Pillow only, mirrors the trainer's exact polygon rasterization and round-based band height, checks the two expected manifest hashes and current recorded config, and saves `results.json` plus stdout in `measure.log`. JSON includes all per-region counts/fractions, normalized spatial masses, aggregate versus uniform-seed means, union coverage, and SHA256 provenance for the script/trainers/config. No images, generator weights, checkpoints or torch modules are loaded.
