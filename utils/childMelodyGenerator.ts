export type NoteDuration = 'w' | 'h' | 'q' | 'e' | 's';

export interface MelodyNote {
  pitch: string;
  duration: NoteDuration;
}

interface GeneratorOptions {
  seed?: number;
  tempo?: number;
  bars?: number;
  ensureLeap?: boolean;
  ensureEighthNotes?: boolean;
}

const scale: string[] = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'E5', 'G5'];

const rhythmPatterns: NoteDuration[][] = [
  ['q', 'q', 'q', 'q'],
  ['q', 'q', 'h'],
  ['h', 'q', 'q'],
  ['h', 'h'],
  ['e', 'e', 'q', 'q', 'q'],
  ['q', 'e', 'e', 'q', 'q'],
  ['q', 'q', 'e', 'e', 'q'],
  ['e', 'q', 'e', 'q', 'q'],
];
const rhythmPatternsWithEighth = rhythmPatterns.filter((pattern) => pattern.includes('e'));
const rhythmPatternsWithoutEighth = rhythmPatterns.filter((pattern) => !pattern.includes('e'));

function mulberry32(seed: number): () => number {
  return function random() {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

function pickStrongTone(random: () => number): string {
  const strongScalePitches = scale.filter((pitch) => ['C', 'E', 'G'].includes(pitch[0]));
  return strongScalePitches[Math.floor(random() * strongScalePitches.length)];
}

function clampIndex(index: number): number {
  if (index < 0) return 0;
  if (index >= scale.length) return scale.length - 1;
  return index;
}

function pickNextIndex(currentIndex: number, random: () => number): number {
  const stepWeights = [0, 1, -1, 2, -2];
  const choice = stepWeights[Math.floor(random() * stepWeights.length)];
  return clampIndex(currentIndex + choice);
}

function chooseRhythm(random: () => number, requireEighth: boolean, allowEighths: boolean): NoteDuration[] {
  const options = requireEighth
    ? rhythmPatternsWithEighth
    : allowEighths
      ? rhythmPatterns
      : rhythmPatternsWithoutEighth;
  return options[Math.floor(random() * options.length)];
}

function barToNotes(
  baseIndex: number,
  random: () => number,
  mustUseEighths: boolean,
  allowEighths: boolean,
): { notes: MelodyNote[]; nextIndex: number; usedEighths: boolean } {
  const rhythm = chooseRhythm(random, mustUseEighths, allowEighths);
  const notes: MelodyNote[] = [];
  let currentIndex = baseIndex;

  rhythm.forEach((duration, idx) => {
    if (idx === 0) {
      const strongPitch = pickStrongTone(random);
      currentIndex = clampIndex(scale.indexOf(strongPitch) >= 0 ? scale.indexOf(strongPitch) : baseIndex);
      notes.push({ pitch: scale[currentIndex], duration });
      return;
    }

    currentIndex = pickNextIndex(currentIndex, random);
    notes.push({ pitch: scale[currentIndex], duration });
  });

  return { notes, nextIndex: currentIndex, usedEighths: rhythm.some((value) => value === 'e') };
}

function injectDecorativeLeap(melody: MelodyNote[], random: () => number): void {
  const candidates = melody
    .map((note, idx) => ({ idx, scaleIndex: scale.indexOf(note.pitch) }))
    .filter(({ idx, scaleIndex }) => idx < melody.length - 2 && scaleIndex >= 0 && scaleIndex < scale.length - 2);

  if (candidates.length === 0) return;

  const choice = candidates[Math.floor(random() * candidates.length)];
  const leapTargets = [choice.scaleIndex + 2, choice.scaleIndex + 3]
    .map((target) => clampIndex(target))
    .filter((target) => target > choice.scaleIndex);

  if (leapTargets.length === 0) return;

  const targetIndex = leapTargets[Math.floor(random() * leapTargets.length)];
  melody[choice.idx].pitch = scale[targetIndex];

  const landingIndex = Math.max(0, targetIndex - 1);
  melody[choice.idx + 1].pitch = scale[landingIndex];
}

export function generateChildMelody({
  seed,
  tempo = 90,
  bars = 4,
  ensureLeap = true,
  ensureEighthNotes = true,
}: GeneratorOptions = {}): { melody: string; tempo: number } {
  const random = mulberry32(seed ?? Date.now());
  let currentIndex = 0;
  const melody: MelodyNote[] = [];
  let usedEighths = false;
  const allowEighths = ensureEighthNotes;

  for (let bar = 0; bar < bars; bar++) {
    const barsRemaining = bars - bar;
    const mustUseEighths = ensureEighthNotes
      ? (!usedEighths && barsRemaining === 1) || (!usedEighths && random() > 0.5) || (usedEighths && random() > 0.2)
      : false;

    const { notes, nextIndex, usedEighths: usedThisBar } = barToNotes(
      currentIndex,
      random,
      mustUseEighths,
      allowEighths,
    );
    melody.push(...notes);
    currentIndex = nextIndex;
    usedEighths = usedEighths || usedThisBar;
  }

  if (ensureLeap) {
    injectDecorativeLeap(melody, random);
  }

  melody[melody.length - 1] = { pitch: 'C4', duration: melody[melody.length - 1].duration };

  const melodyLine = melody.map((note) => `${note.pitch} ${note.duration}`).join(', ');
  return { melody: melodyLine, tempo };
}
