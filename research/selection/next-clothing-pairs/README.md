# Next clothing-teacher shortlist

Original PNGs only were inspected, without model scoring, generation or trainer changes. All five proposed batch2 sources appear clearly adult, masculine and bareheaded, with no obvious additional people and plausible native face anatomy. This is a **clothing-teacher** selection: several are not subjective hot candidates, and none is approved as a final site output.

`training-contact.png` is a labeled diagnostic; the exact1024px original paths/hashes and reasons are in `selection.json`. The unused v1 sources are reserved for possible edited validation rather than new training. No current pair or v1-004 (age uncertain) is selected.

| Source | Priority | Reason and limitation |
| --- | --- | --- |
| b2-042 | Priority | Bald head, glasses, broad smile, angled pose and curved neckline; preserve natural teeth and exact face. Root reviewed suitable as a clothing teacher. |
| b2-058 | Priority | Grey mustache, receding hair, serious three-quarter pose, jacket/shirt/tie; limited but visible front collar space. |
| b2-059 | Priority; **reserved for root edit** | Short grey hair, mustache, glasses, smile and exposed lower neck. Root is generating this edit; do not duplicate. |
| b2-060 | Initially conditional on tight neck; root reviewed suitable | Neutral expression, glasses and existing high white shirt under dark jacket provide a different garment context. Preserve jaw/crop. |
| b2-050 | Lower-priority conditional | Bald head, glasses, gentle smile and textured shirt; little front neck under chin, so only a partial collar may fit. |

Root's native review accepts042/059/060 for research clothing teachers, not hot-output approval. I recommend042 and058 next for variation, with060 also reasonable following root review. There is no forced sixth source:040's chin is cropped,013 adds a cheek microphone, and021's sunglasses conceal eye anatomy. The set is biased toward mature faces and glasses; do not present it as comprehensive diversity or an attractiveness training set.

Native-reviewed v1-017, v1-024 and v1-025 are listed separately in `v1_reserved_validation` and `reserved-validation-contact.png`. They give a full-beard/glasses case, curly grey hair/stubble, and a short-haired three-quarter face without glasses. The017 beard leaves particularly little room for a collar. Their latent IDs must remain outside future training if used for validation.

`prepare.py` packages the manual judgments, verifies source hashes against manual96, and draws diagnostic thumbnails only. It does not modify original pixels or latent files. `selection.json` is not a training manifest or authority to generate images; follow the explicit owner assignment to prevent duplicate edits. No imagegen call was made by this shortlist task.
