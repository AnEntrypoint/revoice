const fs = require('fs');
const manifest = JSON.parse(fs.readFileSync('C:/dev/resound/weights/enhancer_stage2.manifest.json', 'utf8'));
const keys = Object.keys(manifest).sort();
const prefix = process.argv[2] || '';
const exclude = process.argv.slice(3);
let filtered = keys.filter(k => k.startsWith(prefix));
for (const ex of exclude) {
  filtered = filtered.filter(k => !k.includes(ex));
}
for (const k of filtered) {
  console.log(k, JSON.stringify(manifest[k]));
}
console.log('count=' + filtered.length);
