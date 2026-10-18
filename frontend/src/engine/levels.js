// High-Density Level Presets and Layout Generators for Block Breaker RL

export const BRICK_TYPES = {
  EMPTY: 0,
  NORMAL_BLUE: 1,    // 1-hit basic (Cyan, 10 pts)
  NORMAL_GREEN: 2,   // 1-hit basic (Green, 20 pts)
  NORMAL_YELLOW: 3,  // 1-hit basic (Yellow, 30 pts)
  REINFORCED: 4,     // 2-hit brick (Magenta/Pink, 50 pts)
  UNBREAKABLE: 5,    // Steel / Unbreakable (reflects only, 0 pts)
};

export const BRICK_COLORS = {
  [BRICK_TYPES.NORMAL_BLUE]: { fill: '#00f0ff', stroke: '#0284c7', glow: 'rgba(0, 240, 255, 0.65)', hp: 1, points: 10, label: 'Cyan' },
  [BRICK_TYPES.NORMAL_GREEN]: { fill: '#10b981', stroke: '#047857', glow: 'rgba(16, 185, 129, 0.65)', hp: 1, points: 20, label: 'Emerald' },
  [BRICK_TYPES.NORMAL_YELLOW]: { fill: '#f59e0b', stroke: '#b45309', glow: 'rgba(245, 158, 11, 0.65)', hp: 1, points: 30, label: 'Amber' },
  [BRICK_TYPES.REINFORCED]: { fill: '#ec4899', stroke: '#be185d', glow: 'rgba(236, 72, 153, 0.65)', hp: 2, points: 50, label: 'Reinforced' },
  [BRICK_TYPES.UNBREAKABLE]: { fill: '#64748b', stroke: '#334155', glow: 'rgba(100, 116, 139, 0.45)', hp: 999, points: 0, label: 'Steel' },
};

export const LEVEL_PRESETS = {
  invader: {
    id: 'invader',
    name: 'Space Invader Army (Trained)',
    shortName: 'Invader (Trained)',
    badge: 'TRAINED',
    badgeColor: 'text-cyan-400 bg-cyan-500/15 border-cyan-500/40',
    description: 'Standard 88-brick Space Invader configuration used during all 4,000 training episodes.',
    isTrained: true,
    rows: 8,
    cols: 16,
    grid: [
      [0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 4, 0, 0],
      [0, 0, 0, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 0, 0, 0],
      [0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
      [0, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 3, 3, 0],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [2, 2, 0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2],
      [2, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 2],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
  },
  fortress: {
    id: 'fortress',
    name: 'Cyber Fortress (Zero-Shot Test)',
    shortName: 'Fortress (Test)',
    badge: 'UNSEEN TEST',
    badgeColor: 'text-amber-400 bg-amber-500/15 border-amber-500/40',
    description: 'Unseen testing layout featuring twin battlements, steel gateway, and open vertical corridors. Evaluates agent generalization.',
    isTrained: false,
    rows: 8,
    cols: 16,
    grid: [
      [4, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 4],
      [4, 4, 0, 4, 4, 0, 3, 3, 3, 3, 0, 4, 4, 0, 4, 4],
      [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
      [3, 0, 3, 3, 0, 3, 5, 0, 0, 5, 3, 0, 3, 3, 0, 3],
      [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
      [2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2],
      [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
  },
  diamond: {
    id: 'diamond',
    name: 'Diamond Citadel (Zero-Shot Test)',
    shortName: 'Diamond (Test)',
    badge: 'UNSEEN TEST',
    badgeColor: 'text-purple-400 bg-purple-500/15 border-purple-500/40',
    description: 'Unseen geometric diamond formation with open side channels. Tests whether the agent can tunnel and slice along perimeter walls.',
    isTrained: false,
    rows: 8,
    cols: 16,
    grid: [
      [0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 4, 4, 3, 3, 4, 4, 0, 0, 0, 0, 0],
      [0, 0, 0, 4, 4, 3, 3, 5, 5, 3, 3, 4, 4, 0, 0, 0],
      [0, 0, 4, 4, 3, 3, 2, 2, 2, 2, 3, 3, 4, 4, 0, 0],
      [0, 4, 4, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 0],
      [0, 0, 4, 4, 3, 3, 2, 2, 2, 2, 3, 3, 4, 4, 0, 0],
      [0, 0, 0, 4, 4, 3, 3, 5, 5, 3, 3, 4, 4, 0, 0, 0],
      [0, 0, 0, 0, 0, 4, 4, 1, 1, 4, 4, 0, 0, 0, 0, 0],
    ],
  },
};

export function createBrickGrid(levelKey = 'invader') {
  const preset = LEVEL_PRESETS[levelKey] || LEVEL_PRESETS.invader;
  const rawGrid = preset.generate ? preset.generate() : preset.grid;
  const rows = rawGrid.length;
  const cols = rawGrid[0].length;

  const bricks = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const type = rawGrid[r][c];
      const maxHp = type === BRICK_TYPES.REINFORCED ? 2 : (type === BRICK_TYPES.UNBREAKABLE ? 999 : (type === BRICK_TYPES.EMPTY ? 0 : 1));
      row.push({
        type,
        hp: maxHp,
        maxHp,
        row: r,
        col: c,
        points: BRICK_COLORS[type]?.points || 0,
        intact: type !== BRICK_TYPES.EMPTY,
      });
    }
    bricks.push(row);
  }
  return { bricks, rows, cols, presetName: preset.name, preset };
}
