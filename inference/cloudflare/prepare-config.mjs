import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const bundle = new URL('../../models/production/', import.meta.url);
const metadata = JSON.parse(readFileSync(new URL('model.json', bundle), 'utf8'));
const digest = createHash('sha256').update(readFileSync(new URL('generator.safetensors', bundle))).digest('hex');
if (metadata.schema_version !== 1 || !(metadata.training_step > 0)
    || metadata.weights_sha256 !== digest || metadata.review?.approved !== true
    || metadata.review.weights_sha256 !== digest) {
  throw new Error('Refusing an untrained, unreviewed, or mismatched model');
}
const config = JSON.parse(readFileSync(new URL('./wrangler.example.json', import.meta.url), 'utf8'));
config.vars.MODEL_SHA256 = digest;
writeFileSync(new URL('./wrangler.generated.json', import.meta.url), JSON.stringify(config, null, 2) + '\n');
console.log(`Prepared reviewed model ${digest}`);
