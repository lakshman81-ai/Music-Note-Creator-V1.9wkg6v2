<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1dIUiaWPzeUtIcqS3yqfUNoPY4dLgdbnk

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Child-friendly melody helper

Use the new `generateChildMelody` utility (in `utils/childMelodyGenerator.ts`) to quickly draft a 4-bar C major melody tailored for kids. Example usage:

```ts
import { generateChildMelody } from './utils/childMelodyGenerator';

const { melody, tempo } = generateChildMelody({ seed: 42 });
// melody: "C4 q, D4 q, E4 h, ..." and tempo defaults to 90 bpm
// phrases now cadence to C4 and use only pitches drawn from the helper's C-major scale
```

Options you can tweak:

- `bars` (default 4): change the number of bars in the generated phrase.
- `ensureLeap` (default true): inject a single decorative leap that immediately resolves stepwise.
- `ensureEighthNotes` (default true): guarantee at least one bar contains eighth-note motion.

Quick benchmark (requires `ts-node` or similar):

```bash
npx ts-node scripts/benchmarkChildMelody.ts 30
```
