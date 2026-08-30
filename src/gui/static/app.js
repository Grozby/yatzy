/**
 * Yatzy AI Studio - Frontend Application Logic
 * Interactive gameplay, real-time RL suggestions, and autonomous agent play.
 */

// ============================================================================
// Sound Synthesis Engine (Web Audio API - Zero External Dependencies)
// ============================================================================
class SoundEngine {
  constructor() {
    this.ctx = null;
    this.enabled = localStorage.getItem('yatzy_sound_enabled') !== 'false';
  }

  init() {
    if (!this.ctx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioContext();
    }
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggle() {
    this.enabled = !this.enabled;
    localStorage.setItem('yatzy_sound_enabled', this.enabled);
    return this.enabled;
  }

  playClick() {
    if (!this.enabled) return;
    this.init();
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(440, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(120, this.ctx.currentTime + 0.05);
    gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.05);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.05);
  }

  playRoll() {
    if (!this.enabled) return;
    this.init();
    // Simulate dice tumbling noise
    const duration = 0.28;
    const bufferSize = this.ctx.sampleRate * duration;
    const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
    const output = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      output[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.4));
    }
    const whiteNoise = this.ctx.createBufferSource();
    whiteNoise.buffer = buffer;

    const filter = this.ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.setValueAtTime(800, this.ctx.currentTime);
    filter.Q.setValueAtTime(2, this.ctx.currentTime);

    const gain = this.ctx.createGain();
    gain.gain.setValueAtTime(0.4, this.ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + duration);

    whiteNoise.connect(filter);
    filter.connect(gain);
    gain.connect(this.ctx.destination);
    whiteNoise.start();
  }

  playScore() {
    if (!this.enabled) return;
    this.init();
    const now = this.ctx.currentTime;
    [523.25, 659.25].forEach((freq, i) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now + i * 0.08);
      gain.gain.setValueAtTime(0.18, now + i * 0.08);
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.25);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now + i * 0.08);
      osc.stop(now + i * 0.08 + 0.25);
    });
  }

  playBonus() {
    if (!this.enabled) return;
    this.init();
    const notes = [523.25, 659.25, 783.99, 1046.50]; // C5, E5, G5, C6
    const now = this.ctx.currentTime;
    notes.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(freq, now + idx * 0.1);
      gain.gain.setValueAtTime(0.25, now + idx * 0.1);
      gain.gain.exponentialRampToValueAtTime(0.01, now + idx * 0.1 + 0.4);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now + idx * 0.1);
      osc.stop(now + idx * 0.1 + 0.4);
    });
  }

  playGameOver() {
    if (!this.enabled) return;
    this.init();
    const notes = [440, 554.37, 659.25, 880];
    const now = this.ctx.currentTime;
    notes.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, now + idx * 0.12);
      gain.gain.setValueAtTime(0.25, now + idx * 0.12);
      gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.12 + 0.6);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(now + idx * 0.12);
      osc.stop(now + idx * 0.12 + 0.6);
    });
  }
}

const sound = new SoundEngine();

// ============================================================================
// Application State & Controller
// ============================================================================
const state = {
  // Game State
  dices: [1, 1, 1, 1, 1],
  current_roll: 0,
  max_rolls: 3,
  rolls_remaining: 2,
  turn_number: 0,
  total_turns: 15,
  current_action_type: 'SELECT_DICE',
  categories: [],
  upper_section_sum: 0,
  upper_section_threshold: 63,
  has_upper_bonus: false,
  total_score: 0,
  is_game_over: false,

  // AI Suggestions
  suggestions: {
    expected_value: 0,
    dice_keep_probs: [0.5, 0.5, 0.5, 0.5, 0.5],
    recommended_keep_mask: [false, false, false, false, false],
    category_rankings: [],
    best_category: null,
    explanation: 'Ready to play.',
  },

  // UI Selection
  heldDice: [false, false, false, false, false],
  bestScore: parseInt(localStorage.getItem('yatzy_high_score') || '0', 10),

  // Auto Mode
  isAutoPlaying: false,
  autoPlayInterval: null,
  autoPlaySpeed: 700,
};

// ============================================================================
// DOM Element References
// ============================================================================
const dom = {
  // Header
  checkpointSelect: document.getElementById('checkpointSelect'),
  bestScoreDisplay: document.getElementById('bestScoreDisplay'),
  soundToggleBtn: document.getElementById('soundToggleBtn'),
  soundIcon: document.getElementById('soundIcon'),
  newGameBtn: document.getElementById('newGameBtn'),

  // Status Banner
  turnDisplay: document.getElementById('turnDisplay'),
  phaseDisplay: document.getElementById('phaseDisplay'),
  rollsDots: document.getElementById('rollsDots'),
  totalScoreHeader: document.getElementById('totalScoreHeader'),

  // Dice Area
  diceContainer: document.getElementById('diceContainer'),
  diceTip: document.getElementById('diceTip'),
  keepAllBtn: document.getElementById('keepAllBtn'),
  clearKeptBtn: document.getElementById('clearKeptBtn'),
  applyAiKeepBtn: document.getElementById('applyAiKeepBtn'),
  rollDiceBtn: document.getElementById('rollDiceBtn'),
  rollBtnText: document.getElementById('rollBtnText'),
  applyAiActionBtn: document.getElementById('applyAiActionBtn'),
  aiActionBtnText: document.getElementById('aiActionBtnText'),

  // Auto Agent Controls
  autoStepBtn: document.getElementById('autoStepBtn'),
  autoPlayToggleBtn: document.getElementById('autoPlayToggleBtn'),
  autoPlayIcon: document.getElementById('autoPlayIcon'),
  autoPlayText: document.getElementById('autoPlayText'),
  speedSlider: document.getElementById('speedSlider'),
  speedLabel: document.getElementById('speedLabel'),
  speedChips: document.querySelectorAll('.speed-chip'),

  // AI Insights
  expectedScoreValue: document.getElementById('expectedScoreValue'),
  aiExplanationText: document.getElementById('aiExplanationText'),
  aiDiceProbsContainer: document.getElementById('aiDiceProbsContainer'),

  // Scorecard
  scorecardTable: document.getElementById('scorecardTable'),
  scorecardHint: document.getElementById('scorecardHint'),
  upperSumScore: document.getElementById('upperSumScore'),
  bonusProgressBar: document.getElementById('bonusProgressBar'),
  bonusRow: document.getElementById('bonusRow'),
  bonusScore: document.getElementById('bonusScore'),
  bonusStatus: document.getElementById('bonusStatus'),
  grandTotalScore: document.getElementById('grandTotalScore'),
  totalRankText: document.getElementById('totalRankText'),

  // History Log
  historyLog: document.getElementById('historyLog'),
  clearLogBtn: document.getElementById('clearLogBtn'),

  // Modal
  gameOverModal: document.getElementById('gameOverModal'),
  modalFinalScore: document.getElementById('modalFinalScore'),
  modalUpperScore: document.getElementById('modalUpperScore'),
  modalBonusStatus: document.getElementById('modalBonusStatus'),
  modalBestScore: document.getElementById('modalBestScore'),
  modalPerformanceFeedback: document.getElementById('modalPerformanceFeedback'),
  modalNewGameBtn: document.getElementById('modalNewGameBtn'),
};

// ============================================================================
// API Client
// ============================================================================
async function apiRequest(endpoint, method = 'GET', body = null) {
  try {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
    };
    if (body) options.body = JSON.stringify(body);
    const res = await fetch(endpoint, options);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'API request failed');
    }
    return await res.json();
  } catch (error) {
    console.error(`API Error on ${endpoint}:`, error);
    showToast(error.message, 'error');
    throw error;
  }
}

// ============================================================================
// Rendering Functions
// ============================================================================

/**
 * Render standard 3D pips for a die value 1-6.
 */
function createDieHtml(value) {
  let pips = '';
  switch (value) {
    case 1:
      pips = '<div class="pip center"></div>';
      break;
    case 2:
      pips = '<div class="pip top-left"></div><div class="pip bottom-right"></div>';
      break;
    case 3:
      pips = '<div class="pip top-left"></div><div class="pip center"></div><div class="pip bottom-right"></div>';
      break;
    case 4:
      pips = '<div class="pip top-left"></div><div class="pip top-right"></div><div class="pip bottom-left"></div><div class="pip bottom-right"></div>';
      break;
    case 5:
      pips = '<div class="pip top-left"></div><div class="pip top-right"></div><div class="pip center"></div><div class="pip bottom-left"></div><div class="pip bottom-right"></div>';
      break;
    case 6:
      pips = '<div class="pip top-left"></div><div class="pip top-right"></div><div class="pip mid-left"></div><div class="pip mid-right"></div><div class="pip bottom-left"></div><div class="pip bottom-right"></div>';
      break;
    default:
      pips = '<div class="pip center"></div>';
  }
  return `<div class="die value-${value}">${pips}</div>`;
}

/**
 * Update the 5 interactive dice on screen.
 */
function renderDice() {
  dom.diceContainer.innerHTML = '';
  const isDicePhase = state.current_action_type === 'SELECT_DICE' && !state.is_game_over;
  const keepMask = state.suggestions.recommended_keep_mask || [false, false, false, false, false];
  const keepProbs = state.suggestions.dice_keep_probs || [0.5, 0.5, 0.5, 0.5, 0.5];

  state.dices.forEach((val, idx) => {
    const slot = document.createElement('div');
    slot.className = 'die-slot';
    slot.dataset.index = idx;

    const isHeld = state.heldDice[idx];
    const isAiKeep = keepMask[idx];
    const keepProb = Math.round(keepProbs[idx] * 100);

    if (isHeld) slot.classList.add('held');
    if (isDicePhase && isAiKeep) slot.classList.add('ai-recommended');

    // AI Suggestion Pill
    let aiPill = '';
    if (isDicePhase) {
      if (isAiKeep) {
        aiPill = `<span class="ai-suggest-pill" title="AI recommends KEEP (${keepProb}%)">Keep ${keepProb}%</span>`;
      } else {
        aiPill = `<span class="ai-suggest-pill" style="background: rgba(148, 163, 184, 0.85); color: #0f172a;" title="AI recommends REROLL">Reroll</span>`;
      }
    }

    // Status Badge below die
    const badgeText = isHeld ? 'HELD' : 'ROLL';
    const badgeClass = isHeld ? 'badge-held' : 'badge-reroll';
    const statusBadge = `<span class="die-badge ${badgeClass}">${badgeText}</span>`;

    slot.innerHTML = `
      ${aiPill}
      ${createDieHtml(val)}
      ${statusBadge}
    `;

    // Click handler for toggling die keep state
    slot.addEventListener('click', () => {
      if (!isDicePhase) return;
      state.heldDice[idx] = !state.heldDice[idx];
      sound.playClick();
      renderDice();
      updatePlayerActionButtons();
    });

    dom.diceContainer.appendChild(slot);
  });

  // Update quick action buttons
  dom.keepAllBtn.disabled = !isDicePhase;
  dom.clearKeptBtn.disabled = !isDicePhase;
  dom.applyAiKeepBtn.disabled = !isDicePhase;
}

/**
 * Animate rolling dice with tumbling effect.
 */
function triggerDiceRollAnimation() {
  const diceEls = dom.diceContainer.querySelectorAll('.die');
  diceEls.forEach((die, idx) => {
    if (!state.heldDice[idx]) {
      die.classList.add('rolling');
    }
  });
}

/**
 * Render Header & Status Banner.
 */
function renderStatusBanner() {
  dom.turnDisplay.textContent = `${state.turn_number + 1} / ${state.total_turns}`;
  dom.totalScoreHeader.textContent = state.total_score;
  dom.bestScoreDisplay.textContent = state.bestScore || '--';

  // Phase badge
  if (state.is_game_over) {
    dom.phaseDisplay.textContent = '🎉 Game Finished';
    dom.phaseDisplay.className = 'phase-badge badge-category';
  } else if (state.current_action_type === 'SELECT_DICE') {
    dom.phaseDisplay.textContent = `Roll / Keep (Roll ${state.current_roll + 1}/3)`;
    dom.phaseDisplay.className = 'phase-badge badge-roll';
  } else {
    dom.phaseDisplay.textContent = 'Select Category to Score';
    dom.phaseDisplay.className = 'phase-badge badge-category';
  }

  // Roll dots
  dom.rollsDots.innerHTML = '';
  for (let i = 0; i < state.max_rolls; i++) {
    const dot = document.createElement('span');
    dot.className = 'roll-dot';
    // If SELECT_CATEGORY, 0 rolls left. If SELECT_DICE, remaining rolls are active
    if (state.current_action_type === 'SELECT_DICE' && i < state.rolls_remaining) {
      dot.classList.add('active');
    }
    dom.rollsDots.appendChild(dot);
  }

  // Subtitle / Tip
  if (state.current_action_type === 'SELECT_DICE') {
    dom.diceTip.textContent = 'Click dice to Keep or Re-roll, then Roll Dice.';
  } else {
    dom.diceTip.textContent = 'Max rolls reached or all kept. Select a scorecard category on the right.';
  }
}

/**
 * Update the Action Buttons (Roll / Apply AI).
 */
function updatePlayerActionButtons() {
  const isDicePhase = state.current_action_type === 'SELECT_DICE' && !state.is_game_over;
  const isCategoryPhase = state.current_action_type === 'SELECT_CATEGORY' && !state.is_game_over;

  if (isDicePhase) {
    dom.rollDiceBtn.disabled = false;
    dom.rollBtnText.textContent = `Roll Dice (${state.rolls_remaining} Left)`;
    dom.applyAiActionBtn.disabled = false;
    dom.aiActionBtnText.textContent = 'Apply AI Move (Roll/Keep)';
  } else if (isCategoryPhase) {
    dom.rollDiceBtn.disabled = true;
    dom.rollBtnText.textContent = 'Must Select Category';
    dom.applyAiActionBtn.disabled = false;
    const bestCatName = state.suggestions.best_category ? state.suggestions.best_category.display_name : 'Best Category';
    dom.aiActionBtnText.textContent = `Score AI Pick: ${bestCatName}`;
  } else {
    dom.rollDiceBtn.disabled = true;
    dom.rollBtnText.textContent = 'Game Over';
    dom.applyAiActionBtn.disabled = true;
    dom.aiActionBtnText.textContent = 'Game Complete';
  }
}

/**
 * Render the Scorecard Table with AI recommendations & hover previews.
 */
function renderScorecard() {
  const isCategoryPhase = state.current_action_type === 'SELECT_CATEGORY' && !state.is_game_over;
  const rankings = state.suggestions.category_rankings || [];
  const rankingMap = {};
  rankings.forEach((r) => {
    rankingMap[r.index] = r;
  });

  state.categories.forEach((cat) => {
    const row = document.getElementById(`row-${cat.index}`);
    const scoreCell = document.getElementById(`score-${cat.index}`);
    const metaCell = document.getElementById(`meta-${cat.index}`);
    if (!row || !scoreCell || !metaCell) return;

    // Reset classes
    row.className = 'cat-row';
    const aiInfo = rankingMap[cat.index];

    if (cat.is_filled) {
      row.classList.add('filled');
      scoreCell.textContent = cat.score;
      metaCell.innerHTML = `<span style="color: var(--text-dim); font-size: 0.8rem;">✓ Scored</span>`;
      row.onclick = null;
    } else {
      scoreCell.textContent = '-';
      let metaHtml = '';

      if (isCategoryPhase && aiInfo) {
        row.classList.add('clickable');
        const prob = Math.round(aiInfo.probability * 100);
        const pts = cat.potential_score;

        if (aiInfo.rank === 1) {
          row.classList.add('best-ai-pick');
          metaHtml = `
            <span class="ai-badge ai-badge-top" title="AI Confidence: ${prob}%">
              ★ #1 Pick (${prob}%) <strong class="preview-pts">+${pts}</strong>
            </span>`;
        } else if (aiInfo.rank <= 3 && prob >= 5) {
          metaHtml = `
            <span class="ai-badge ai-badge-secondary" title="Rank #${aiInfo.rank}">
              #${aiInfo.rank} (${prob}%) <strong class="preview-pts">+${pts}</strong>
            </span>`;
        } else {
          metaHtml = `<span class="preview-pts" style="opacity: 0.7;">+${pts} pts</span>`;
        }

        // On click: score this category
        row.onclick = () => {
          handleCategoryClick(cat.index);
        };
      } else {
        // SELECT_DICE phase: show subtle current potential score
        const pts = cat.potential_score;
        metaHtml = `<span class="preview-pts" style="color: var(--text-dim); font-size: 0.8rem;">Potential: ${pts}</span>`;
        row.onclick = null;
      }

      metaCell.innerHTML = metaHtml;
    }
  });

  // Upper Summary & Bonus Progress
  const upperSum = state.upper_section_sum;
  const threshold = state.upper_section_threshold;
  dom.upperSumScore.textContent = `${upperSum} / ${threshold}`;

  const pct = Math.min(100, Math.round((upperSum / threshold) * 100));
  dom.bonusProgressBar.style.width = `${pct}%`;

  if (state.has_upper_bonus) {
    dom.bonusScore.textContent = '50';
    dom.bonusScore.className = 'cat-score highlight-emerald';
    dom.bonusStatus.innerHTML = '<span class="highlight-emerald" style="font-weight: 700;">✓ ACHIEVED (+50)</span>';
  } else {
    dom.bonusScore.textContent = '0';
    dom.bonusScore.className = 'cat-score';
    const remainingNeeded = Math.max(0, threshold - upperSum);
    dom.bonusStatus.textContent = `Needs ${remainingNeeded} more`;
  }

  // Grand Total
  dom.grandTotalScore.textContent = state.total_score;
  if (state.is_game_over) {
    dom.totalRankText.innerHTML = '<strong class="highlight-gold">Final Total</strong>';
  } else {
    dom.totalRankText.textContent = `Turn ${state.turn_number + 1} of 15`;
  }

  // Scorecard hint
  if (isCategoryPhase) {
    dom.scorecardHint.innerHTML = '<span class="highlight-gold" style="font-weight: 700;">👉 Choose a category to lock in your score</span>';
  } else {
    dom.scorecardHint.textContent = 'Roll phase active';
  }
}

/**
 * Render Model Insights & Expected Value.
 */
function renderInsights() {
  const sug = state.suggestions;
  dom.expectedScoreValue.textContent = sug.expected_value ? `${Math.round(sug.expected_value)} pts` : '--';
  dom.aiExplanationText.textContent = sug.explanation || 'Model analyzing situation...';

  // Render dice keep probabilities bar
  dom.aiDiceProbsContainer.innerHTML = '';
  if (state.current_action_type === 'SELECT_DICE' && sug.dice_keep_probs) {
    sug.dice_keep_probs.forEach((prob, idx) => {
      const pct = Math.round(prob * 100);
      const isKeep = sug.recommended_keep_mask[idx];
      const val = state.dices[idx];
      const div = document.createElement('div');
      div.className = 'die-prob-item';
      div.style.borderColor = isKeep ? 'var(--color-cyan)' : 'var(--surface-border)';
      div.innerHTML = `
        <span style="color: var(--text-dim); font-size: 0.7rem;">Die ${idx + 1} (${val})</span>
        <span class="die-prob-val" style="color: ${isKeep ? 'var(--color-cyan)' : 'var(--text-muted)'};">
          ${isKeep ? 'KEEP' : 'REROLL'} (${pct}%)
        </span>
      `;
      dom.aiDiceProbsContainer.appendChild(div);
    });
  }
}

/**
 * Render the History / Game Event Feed.
 */
function renderHistory(events = []) {
  if (!events || events.length === 0) return;
  dom.historyLog.innerHTML = '';
  events.slice().reverse().forEach((ev) => {
    const item = document.createElement('div');
    item.className = `log-item ${ev.type || 'roll'}`;
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    item.innerHTML = `
      <span>${ev.message}</span>
      <span style="font-size: 0.7rem; color: var(--text-dim); margin-left: 0.5rem;">${timeStr}</span>
    `;
    dom.historyLog.appendChild(item);
  });
}

/**
 * Synchronize full state payload from backend.
 */
function applyServerState(fullPayload) {
  const { state: s, suggestions: sug, last_step_info: lastInfo, game_history: hist } = fullPayload;

  // Update local state
  state.dices = s.dices;
  state.current_roll = s.current_roll;
  state.max_rolls = s.max_rolls;
  state.rolls_remaining = s.rolls_remaining;
  state.turn_number = s.turn_number;
  state.total_turns = s.total_turns;
  state.current_action_type = s.current_action_type;
  state.categories = s.categories;
  state.upper_section_sum = s.upper_section_sum;
  state.upper_section_threshold = s.upper_section_threshold;
  state.has_upper_bonus = s.has_upper_bonus;
  state.total_score = s.total_score;
  state.is_game_over = s.is_game_over;

  state.suggestions = sug || state.suggestions;

  // On new turn / reset held dice if category was just scored
  if (state.current_action_type === 'SELECT_DICE' && state.current_roll === 0) {
    state.heldDice = [false, false, false, false, false];
  }

  // Check personal best
  if (state.total_score > state.bestScore) {
    state.bestScore = state.total_score;
    localStorage.setItem('yatzy_high_score', state.bestScore.toString());
  }

  // Render UI
  renderStatusBanner();
  renderDice();
  renderScorecard();
  renderInsights();
  updatePlayerActionButtons();
  if (hist) renderHistory(hist);

  // Check Game Over
  if (state.is_game_over) {
    stopAutoPlay();
    showGameOverModal();
  }
}

// ============================================================================
// Gameplay Actions
// ============================================================================

/**
 * Handle Roll Dice button click.
 */
async function handleRollDice() {
  if (state.current_action_type !== 'SELECT_DICE' || state.is_game_over) return;
  sound.playRoll();
  triggerDiceRollAnimation();

  try {
    const res = await apiRequest('/api/game/roll', 'POST', { keep_mask: state.heldDice });
    setTimeout(() => {
      applyServerState(res.data);
    }, 280);
  } catch (err) {
    console.error(err);
  }
}

/**
 * Handle Category Click in Category Selection Phase.
 */
async function handleCategoryClick(catIndex) {
  if (state.current_action_type !== 'SELECT_CATEGORY' || state.is_game_over) return;
  sound.playScore();

  try {
    const res = await apiRequest('/api/game/select-category', 'POST', { category_index: catIndex });
    if (res.data.last_step_info && res.data.last_step_info.bonus_awarded) {
      sound.playBonus();
    }
    applyServerState(res.data);
  } catch (err) {
    console.error(err);
  }
}

/**
 * Execute 1 AI Step (Deterministic RL policy).
 */
async function handleStepAi() {
  if (state.is_game_over) {
    stopAutoPlay();
    return;
  }

  if (state.current_action_type === 'SELECT_DICE') {
    sound.playRoll();
    triggerDiceRollAnimation();
  } else {
    sound.playScore();
  }

  try {
    const res = await apiRequest('/api/game/step-ai', 'POST');
    if (res.data.last_step_info && res.data.last_step_info.bonus_awarded) {
      sound.playBonus();
    }
    applyServerState(res.data);
  } catch (err) {
    stopAutoPlay();
    console.error(err);
  }
}

/**
 * Apply AI Move based on current phase.
 */
async function handleApplyAiMove() {
  if (state.is_game_over) return;

  if (state.current_action_type === 'SELECT_DICE') {
    // Apply recommended keep mask and roll
    state.heldDice = [...state.suggestions.recommended_keep_mask];
    renderDice();
    await handleRollDice();
  } else {
    // Score best category
    if (state.suggestions.best_category) {
      await handleCategoryClick(state.suggestions.best_category.index);
    }
  }
}

/**
 * Start a brand new game session.
 */
async function handleNewGame() {
  stopAutoPlay();
  hideGameOverModal();
  sound.playClick();

  try {
    const res = await apiRequest('/api/game/new', 'POST');
    state.heldDice = [false, false, false, false, false];
    applyServerState(res.data);
  } catch (err) {
    console.error(err);
  }
}

// ============================================================================
// Autonomous Agent Mode (Auto Play Loop)
// ============================================================================

function toggleAutoPlay() {
  if (state.isAutoPlaying) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
}

function startAutoPlay() {
  if (state.is_game_over) {
    handleNewGame().then(() => {
      startAutoPlayLoop();
    });
    return;
  }
  startAutoPlayLoop();
}

function startAutoPlayLoop() {
  state.isAutoPlaying = true;
  dom.autoPlayToggleBtn.classList.replace('btn-primary', 'btn-emerald');
  dom.autoPlayIcon.textContent = '⏸';
  dom.autoPlayText.textContent = 'Pause Agent';

  const runStep = async () => {
    if (!state.isAutoPlaying || state.is_game_over) {
      stopAutoPlay();
      return;
    }
    await handleStepAi();
    if (state.isAutoPlaying && !state.is_game_over) {
      state.autoPlayInterval = setTimeout(runStep, state.autoPlaySpeed);
    }
  };

  state.autoPlayInterval = setTimeout(runStep, 100);
}

function stopAutoPlay() {
  state.isAutoPlaying = false;
  if (state.autoPlayInterval) {
    clearTimeout(state.autoPlayInterval);
    state.autoPlayInterval = null;
  }
  dom.autoPlayToggleBtn.classList.replace('btn-emerald', 'btn-primary');
  dom.autoPlayIcon.textContent = '▶';
  dom.autoPlayText.textContent = 'Auto Play Game';
}

function updateAutoPlaySpeed(speedMs) {
  state.autoPlaySpeed = parseInt(speedMs, 10);
  dom.speedSlider.value = state.autoPlaySpeed;

  let speedText = 'Normal';
  if (state.autoPlaySpeed <= 100) speedText = 'Turbo';
  else if (state.autoPlaySpeed <= 350) speedText = 'Fast';
  else if (state.autoPlaySpeed <= 800) speedText = 'Normal';
  else speedText = 'Slow';

  dom.speedLabel.textContent = `${speedText} (${state.autoPlaySpeed}ms)`;

  // Update chip highlights
  dom.speedChips.forEach((chip) => {
    const chipSpeed = parseInt(chip.dataset.speed, 10);
    chip.classList.toggle('active', Math.abs(chipSpeed - state.autoPlaySpeed) < 100);
  });
}

// ============================================================================
// Checkpoint Selector
// ============================================================================

async function loadCheckpoints() {
  try {
    const data = await apiRequest('/api/checkpoints');
    dom.checkpointSelect.innerHTML = '';

    data.checkpoints.forEach((ckpt) => {
      const opt = document.createElement('option');
      opt.value = ckpt.path;
      opt.textContent = `${ckpt.name} [${ckpt.info}]`;
      if (ckpt.path === data.active_checkpoint) {
        opt.selected = true;
      }
      dom.checkpointSelect.appendChild(opt);
    });

    dom.checkpointSelect.addEventListener('change', async (e) => {
      const chosenPath = e.target.value;
      if (!chosenPath) return;
      try {
        const res = await apiRequest('/api/checkpoints/load', 'POST', { path: chosenPath });
        showToast('Loaded model checkpoint successfully!', 'success');
        applyServerState(res.data);
      } catch (err) {
        console.error(err);
      }
    });
  } catch (err) {
    console.error('Failed to load checkpoints:', err);
  }
}

// ============================================================================
// Game Over Modal & Score Feedback
// ============================================================================

function showGameOverModal() {
  sound.playGameOver();
  dom.modalFinalScore.textContent = state.total_score;
  dom.modalUpperScore.textContent = `${state.upper_section_sum} / 63`;
  dom.modalBonusStatus.textContent = state.has_upper_bonus ? 'Achieved (+50) ✓' : 'Missed (0)';
  dom.modalBonusStatus.className = `stat-box-val ${state.has_upper_bonus ? 'highlight-emerald' : ''}`;
  dom.modalBestScore.textContent = state.bestScore;

  // Performance tier feedback
  const score = state.total_score;
  let feedback = '';
  if (score >= 230) {
    feedback = '🏆 Outstanding game! Top tier Yatzy score!';
  } else if (score >= 180) {
    feedback = '🔥 Great performance! Well above average!';
  } else if (score >= 140) {
    feedback = '👍 Solid game! Good decision making!';
  } else {
    feedback = '🔄 Tough dice rolls! Hit New Game to try again!';
  }
  dom.modalPerformanceFeedback.textContent = feedback;

  dom.gameOverModal.classList.remove('hidden');
}

function hideGameOverModal() {
  dom.gameOverModal.classList.add('hidden');
}

// ============================================================================
// Toast Notification Utility
// ============================================================================
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.style.position = 'fixed';
  toast.style.bottom = '24px';
  toast.style.right = '24px';
  toast.style.padding = '12px 20px';
  toast.style.borderRadius = '8px';
  toast.style.background = type === 'error' ? '#ef4444' : '#10b981';
  toast.style.color = '#ffffff';
  toast.style.fontWeight = '600';
  toast.style.fontSize = '0.9rem';
  toast.style.boxShadow = '0 10px 25px rgba(0,0,0,0.5)';
  toast.style.zIndex = '9999';
  toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
  toast.textContent = message;

  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// ============================================================================
// Event Listeners & Initialization
// ============================================================================

function bindEventListeners() {
  // Primary actions
  dom.rollDiceBtn.addEventListener('click', handleRollDice);
  dom.applyAiActionBtn.addEventListener('click', handleApplyAiMove);
  dom.newGameBtn.addEventListener('click', handleNewGame);
  dom.modalNewGameBtn.addEventListener('click', handleNewGame);

  // Quick dice buttons
  dom.keepAllBtn.addEventListener('click', () => {
    state.heldDice = [true, true, true, true, true];
    sound.playClick();
    renderDice();
  });

  dom.clearKeptBtn.addEventListener('click', () => {
    state.heldDice = [false, false, false, false, false];
    sound.playClick();
    renderDice();
  });

  dom.applyAiKeepBtn.addEventListener('click', () => {
    state.heldDice = [...state.suggestions.recommended_keep_mask];
    sound.playClick();
    renderDice();
    showToast('Applied AI keep recommendations!', 'info');
  });

  // Auto Mode
  dom.autoStepBtn.addEventListener('click', handleStepAi);
  dom.autoPlayToggleBtn.addEventListener('click', toggleAutoPlay);

  // Speed controls
  dom.speedSlider.addEventListener('input', (e) => {
    updateAutoPlaySpeed(e.target.value);
  });

  dom.speedChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      updateAutoPlaySpeed(chip.dataset.speed);
    });
  });

  // Sound toggle
  dom.soundToggleBtn.addEventListener('click', () => {
    const isEnabled = sound.toggle();
    dom.soundIcon.textContent = isEnabled ? '🔊' : '🔇';
    showToast(`Sound ${isEnabled ? 'Enabled' : 'Muted'}`, 'info');
  });
  dom.soundIcon.textContent = sound.enabled ? '🔊' : '🔇';

  // History Clear
  dom.clearLogBtn.addEventListener('click', () => {
    dom.historyLog.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 1rem;">Log cleared</div>';
  });

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    if (e.code === 'Space') {
      e.preventDefault();
      if (state.current_action_type === 'SELECT_DICE') {
        handleRollDice();
      } else {
        handleApplyAiMove();
      }
    } else if (e.code === 'KeyN') {
      handleNewGame();
    } else if (e.code === 'KeyA') {
      toggleAutoPlay();
    } else if (e.code === 'Digit1') toggleDieByIndex(0);
    else if (e.code === 'Digit2') toggleDieByIndex(1);
    else if (e.code === 'Digit3') toggleDieByIndex(2);
    else if (e.code === 'Digit4') toggleDieByIndex(3);
    else if (e.code === 'Digit5') toggleDieByIndex(4);
  });
}

function toggleDieByIndex(idx) {
  if (state.current_action_type !== 'SELECT_DICE' || state.is_game_over) return;
  state.heldDice[idx] = !state.heldDice[idx];
  sound.playClick();
  renderDice();
}

/**
 * Initialize application on page load.
 */
async function initApp() {
  bindEventListeners();
  updateAutoPlaySpeed(700);

  // Load available checkpoints
  await loadCheckpoints();

  // Load initial game state
  try {
    const res = await apiRequest('/api/game/state');
    applyServerState(res.data);
  } catch (err) {
    console.error('Failed to fetch initial state:', err);
  }
}

document.addEventListener('DOMContentLoaded', initApp);
