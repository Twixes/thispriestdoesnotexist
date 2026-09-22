"""Small NumPy/JSON fixtures only. Never construct or load a native model."""
import copy
import hashlib
import json
import unittest
import numpy as np
import render
import build


class PreparationTests(unittest.TestCase):
    def test_new_latents_are_reproducible_complete_and_distinct(self):
        old=np.zeros((4,512),dtype=np.float32)
        z=np.random.default_rng(render.SEED).standard_normal((32,512)).astype(np.float32)
        original=render.check_latents(z,old)
        self.assertEqual(original,render.check_latents(z.copy(),old))
        for change in ('duplicate','old_overlap','nan','shape','dtype'):
            bad=z.copy()
            if change=='duplicate':bad[1]=bad[0]
            elif change=='old_overlap':bad[1]=old[0]
            elif change=='nan':bad[1,1]=np.nan
            elif change=='shape':bad=bad[:31]
            else:bad=bad.astype(np.float64)
            with self.subTest(change=change),self.assertRaises(AssertionError):render.check_latents(bad,old)

    def test_checkpoint_envelope_rejects_wrong_counts_masks_and_provenance_before_mutation(self):
        # Strings intentionally stand in for tensor/optimizer values; no torch call occurs.
        state={key:'DECLARED_SYNTHETIC_STATE' for key in ('G','Gema','D','g_optimizer','d_optimizer','torch_rng','sampling_rng','path_mean','controller_enabled','optimizer_parameter_names')}
        state.update(iterations=100,optimizer_steps={'g':125,'d':107},selections={'G':'declared-mask','D':'declared-mask'})
        payload={'engine':state,'protocol_sha256':'protocol','runtime_sha256':'runtime','modulation_inventory_sha256':'inventory','source_main_checkpoint_sha256':'parent'}
        protocol={'selections':copy.deepcopy(state['selections']),'source_main_checkpoint_sha256':'parent'}
        digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
        def evaluate(value,repin=False):
            marker={'state_sha256':digest(value if repin else payload)}
            return render.check_envelope(value,marker,protocol,'protocol','runtime','inventory',digest)
        self.assertEqual(evaluate(payload),state)
        for mutation in ('hash','counts','masks','runtime','parent','missing_gema'):
            bad=copy.deepcopy(payload)
            if mutation in ('hash','counts'):bad['engine']['optimizer_steps']['g']=124
            elif mutation=='masks':bad['engine']['selections']['G']='wrong-mask'
            elif mutation=='runtime':bad['runtime_sha256']='other'
            elif mutation=='parent':bad['source_main_checkpoint_sha256']='other'
            else:del bad['engine']['Gema']
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):evaluate(bad,repin=mutation!='hash')
        self.assertEqual(payload['engine'],state)

    def test_review_refuses_omitted_duplicate_swapped_or_unbound_image_rows(self):
        registry={'row_sha256':[f'declared-latent-{i}' for i in range(32)]}
        manifest={'complete':True,'count':32,'image_count':64,'unfiltered':True,'all_native_whole_images':True,
                  'rng_unchanged':True,'fixed_offset_and_mask_invariants':True,'parent_reproduction_exact':True,
                  'production_approved':False,'server_latency_proven':False,
                  'images':[{'arm':arm,'index':i,'path':f'{arm}-{i:03}.png','width':1024,'height':1024,'latent_sha256':registry['row_sha256'][i]} for arm in ('raw','ema') for i in range(32)]}
        build.validate_coverage(manifest,registry)
        for change in ('missing','duplicate','swapped','latent','dimensions','approval'):
            bad=copy.deepcopy(manifest)
            if change=='missing':bad['images'].pop()
            elif change=='duplicate':bad['images'][1]=bad['images'][0]
            elif change=='swapped':bad['images'][0]['path']='ema-000.png'
            elif change=='latent':bad['images'][0]['latent_sha256']='wrong-latent'
            elif change=='dimensions':bad['images'][0]['width']=512
            else:bad['production_approved']=True
            with self.subTest(change=change),self.assertRaises(AssertionError):build.validate_coverage(bad,registry)
        rows=[{'index':i,'images':{arm:{'status':'pending'} for arm in ('raw','ema')}} for i in range(32)]
        blank=build.annotation_template(rows)
        self.assertEqual(len(blank['rows']),64)
        self.assertTrue(all(row['status']=='not_reviewed' and all(row[f] is None for f in build.FIELDS) for row in blank['rows']))


if __name__=='__main__':unittest.main(verbosity=2)
