# Frozen SDXL-Turbo priest LoRA pilot

The executable specification is `sdxl-pilot-protocol.json`, SHA256 **ac8448d8facd5910d0e512486522a436a8e5360903ac8867bf63d04e7abfb22d**. Freeze that file in every launch record before training/evaluation. This task prepared the protocol and ran tokenizer/file checks; it generated no images and performed no training.

Use the pinned existing SDXL-Turbo revision `71153311d3dbb46851df1931d3ca6e939de83304` and reviewed dataset manifest `6796be940a10610843c154c7063b3bb907d0ffab18e014737c6abc6f02ee340b`. Only the **20 training images** may enter fitting, latent caching or augmentation. The five validation and five test teacher portraits remain held out; these image splits are distinct from generated evaluation seeds.

This is a deliberately small compatibility and adaptation experiment. SDXL-Turbo uses adversarial diffusion distillation for 1–4-step sampling; the official recipe uses guidance zero and favors 512px. Its model card explicitly acknowledges imperfect photorealism and faces. Ordinary noise-prediction LoRA training does not reproduce its distillation objective. The concern that adaptation may undo one-step quality is an engineering inference, not a claim that every LoRA necessarily fails. [Official model card](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/README.md).

Technical preflight amendment: training seed `202609219900` exceeded NumPy’s unsigned32bit range, so it is now `2026092199`. The prior JSON and its hash are preserved in `sdxl-pilot-protocol.pre-seed-amendment.json` and `sdxl-pilot-protocol-amendments.json`. No evaluation prompts, Torch-valid inference seeds, arms or gates changed.

## Bounded training

One recipe: UNet attention LoRA **rank16/alpha16, LR1e-5, AdamW, batch1, constant LR, fp16 base weights with fp32 adapter parameters**, frozen text encoders/VAE, 512 square inputs and gradient checkpointing. Use the per-image captions with `PR1EST_CAL`. No random crop, flip, color jitter, extra images or rank/LR sweep. The JSON specifies optimizer details and the fixed training seed.

Save at **20, 100 and 250 total optimizer updates**. Twenty verifies finite loss/gradients, changed adapter tensors, reloadability and visible outputs; it proves no quality gain. Inspect the full development set at each completed checkpoint. Stop on numerical/resource failure or clear native face-quality/identity-diversity regression. Continue from100 only if the development evidence justifies the remaining bounded pilot. Low loss is insufficient. Select a checkpoint using the frozen development ranking, with earlier checkpoint winning an exact tie.

## Matched evaluation

**24 development cases:** the existing eight comparison prompts/seeds are preserved exactly, with their original saved latent file hashes. Sixteen additional prompts vary apparent age28–79, face shape, hair/baldness, facial hair, skin appearance, frontal/three-quarter pose, expression and background. They avoid restricting every face to the previous four repeatedly prompted families. All new descriptors are requested generation attributes, not identity or ethnicity judgments about dataset subjects. Both local CLIP tokenizers were checked without model computation: every prompt including its trigger fits the77-token limit; maximum75.

For each case compare four arms: base/plain, base/trigger, LoRA/plain and LoRA/trigger. The trigger version is exactly `PR1EST_CAL. ` plus the same prompt. Comparing LoRA/trigger with base/trigger isolates adaptation better than comparing against a base that never received the unfamiliar token. The no-trigger arms expose how the token itself changes the base and whether the adapter broadly alters other prompts.

Use identical saved initial noise and generator state for matched arms. Merely reusing a seed or initial tensor is insufficient for four-step Euler ancestral sampling: subsequent random draws must match too. Reuse legacy latents; generate new float32 CPU noise once and retain it before casting to the device. Re-run base arms under the same pinned current runtime as the adapter; earlier images remain historical evidence. Save raw RGB PNGs, complete prompts, hashes, scheduler/software/device data and timing for **every** output. Retain failures. No selection, retry, repair or post-generation crop. An optional grayscale WebP is a display artifact; inspect the native image as well.

Primary inference is **one step,512×512,guidance0,adapter scale1**. Only if every adapted one-step candidate fails the development gate, compare **four steps** across all completed checkpoints and matched base arms on all24 development cases. Record the one-step failure. No search over two/three steps or adapter strengths. Freeze checkpoint and step count before opening the test results.

## Sealed test and review

**32 disjoint test seeds:** sixteen repeat the varied development prompt texts with unseen latents; another sixteen use one unchanged generic priest prompt. Thus the varied block tests new-seed behavior, not unseen-language generalization. The generic block specifically checks whether the learned domain collapses into a few faces when appearance instructions do not force diversity. Run all four arms for the selected checkpoint/step count only. Test seed assignments already exist in the JSON and must not be replaced.

Every output receives native-size review for facial integrity, photographic texture, recognizable clerical clothing, adult male appearance, no hats and one person. Calendar appeal, pose/age matching, suspected teacher resemblance and visual face families are recorded separately. A legitimate broad white clerical collar is acceptable; failing to render the original prompt's exact small Roman tab is not a user quality failure.

The JSON freezes explicit **research selection heuristics**, not user approval criteria: development needs at least22/24 joint passes,23/24 intact faces, no regression against same-trigger base, no strong teacher copying, at least16 plausible face families and no family larger than4. Calendar-appeal paired wins must at least equal losses. Rank eligible checkpoints by joint passes, intact faces, distinct families and calendar-appeal net wins, then earlier update. Report every outcome and uncertainty. Visual families are conservative manual groupings, not biometric proof.

Test reporting uses30/32 joint passes,31/32 intact faces and no reduction against same-trigger base; the generic block needs at least12/16 plausible families, none larger than3. These small samples cannot establish population performance. After test exposure, do not refine prompts, labels, seeds, adapter strength or checkpoint selection. Any revised experiment needs a new protocol and fresh test seeds while preserving this result.

The teacher set remains small and stylistically narrow: monochrome, dark shirts, mostly tight portraits and simple backgrounds. Few bald faces and limited extreme old age make the new prompts useful stress cases; they cannot manufacture missing training coverage. Passing this pilot only identifies a candidate. Production still requires native output review and **actual complete server generation below500ms**, including encoding and any filtering/retries. Local MPS times do not prove server latency; report cold starts, warm median/p95/max, queueing and public request latency separately.
