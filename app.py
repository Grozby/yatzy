"""
Yatzy AI Studio - Hugging Face ZeroGPU Gradio Application
Interactive Yatzy player with real-time PPO RL move suggestions and autonomous agent play.
"""

import time
from typing import Generator, List, Tuple
import gradio as gr

# Safe import for Hugging Face ZeroGPU
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is None:
                return lambda f: f
            return fn

from src.gui.service import GameService

# Initialize game service
service = GameService()

DICE_EMOJIS = {1: "⚀", 2: "⚁", 3: "⚂", 4: "⚃", 5: "⚄", 6: "⚅"}


def format_dice_html(state: dict, suggestions: dict) -> str:
    """Render a visual tray of the 5 dice with AI keep badges."""
    dices = state.get("dices", [1, 1, 1, 1, 1])
    keep_probs = suggestions.get("dice_keep_probs", [0.5] * 5)
    rec_mask = suggestions.get("recommended_keep_mask", [False] * 5)
    is_game_over = state.get("is_game_over", False)

    cards = []
    for i, d in enumerate(dices):
        is_rec = rec_mask[i] and not is_game_over
        prob = int(keep_probs[i] * 100)
        border_color = "#06b6d4" if is_rec else "#334155"
        bg_color = "#082f49" if is_rec else "#1e293b"
        
        if is_game_over:
            tag = '<span style="background: rgba(148,163,184,0.15); color: #94a3b8; padding: 2px 6px; border-radius: 6px; font-size: 11px;">Final</span>'
        elif is_rec:
            tag = f'<span style="background: rgba(6,182,212,0.25); color: #38bdf8; padding: 2px 6px; border-radius: 6px; font-size: 11px; font-weight: 700;">★ KEEP ({prob}%)</span>'
        else:
            tag = f'<span style="background: rgba(148,163,184,0.15); color: #94a3b8; padding: 2px 6px; border-radius: 6px; font-size: 11px;">REROLL ({prob}%)</span>'

        cards.append(
            f"""
            <div style="display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 12px 16px; background: {bg_color}; border: 2px solid {border_color}; border-radius: 12px; min-width: 90px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                {tag}
                <span style="font-size: 42px; line-height: 1; margin: 4px 0;">{DICE_EMOJIS.get(d, '🎲')}</span>
                <strong style="font-size: 15px; color: #f8fafc;">Die {i+1} : {d}</strong>
            </div>
            """
        )

    return f"""
    <div style="display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; padding: 16px; background: #0b0f19; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);">
        {''.join(cards)}
    </div>
    """


def format_scorecard_html(state: dict, suggestions: dict) -> str:
    """Render classic Yatzy scorecard table with Upper Bonus progress and AI recommendations."""
    categories = state.get("categories", [])
    rankings = suggestions.get("category_rankings", [])
    rank_map = {r["index"]: r for r in rankings}
    is_game_over = state.get("is_game_over", False)

    upper_rows = []
    lower_rows = []

    for cat in categories:
        idx = cat["index"]
        name = cat["display_name"]
        is_filled = cat["is_filled"]
        score = cat["score"]
        potential = cat["potential_score"]
        ai_info = rank_map.get(idx, {})

        if is_filled:
            score_display = f'<strong style="color: #f8fafc; font-family: monospace; font-size: 15px;">{score}</strong>'
            meta_display = '<span style="color: #64748b; font-size: 12px;">✓ Scored</span>'
            row_bg = "rgba(255, 255, 255, 0.02)"
        elif is_game_over:
            score_display = '<span style="color: #64748b; font-family: monospace;">-</span>'
            meta_display = '<span style="color: #64748b; font-size: 12px;">-</span>'
            row_bg = "transparent"
        else:
            prob = int(ai_info.get("probability", 0) * 100)
            rank = ai_info.get("rank", None)
            if rank == 1:
                score_display = f'<span style="color: #34d399; font-weight: 700; font-family: monospace;">(+{potential})</span>'
                meta_display = f'<span style="background: rgba(16,185,129,0.25); color: #34d399; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 12px; border: 1px solid rgba(16,185,129,0.4);">★ #1 AI Pick ({prob}%)</span>'
                row_bg = "rgba(16, 185, 129, 0.12)"
            elif rank and rank <= 3 and prob >= 5:
                score_display = f'<span style="color: #93c5fd; font-family: monospace;">(+{potential})</span>'
                meta_display = f'<span style="background: rgba(59,130,246,0.15); color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 11px;">#{rank} ({prob}%)</span>'
                row_bg = "rgba(59, 130, 246, 0.05)"
            else:
                score_display = f'<span style="color: #64748b; font-family: monospace;">(+{potential})</span>'
                meta_display = f'<span style="color: #64748b; font-size: 12px;">Potential: +{potential}</span>'
                row_bg = "transparent"

        row_html = f"""
        <tr style="background: {row_bg}; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 8px 12px; font-weight: 500;">{name}</td>
            <td style="padding: 8px 12px; text-align: center;">{score_display}</td>
            <td style="padding: 8px 12px; text-align: right;">{meta_display}</td>
        </tr>
        """
        if cat["is_upper"]:
            upper_rows.append(row_html)
        else:
            lower_rows.append(row_html)

    upper_sum = state.get("upper_section_sum", 0)
    has_bonus = state.get("has_upper_bonus", False)
    bonus_text = '<span style="color: #34d399; font-weight: 700;">✓ ACHIEVED (+50 PTS)</span>' if has_bonus else f'<span style="color: #94a3b8;">{max(0, 63 - upper_sum)} more needed</span>'
    total_score = state.get("total_score", 0)
    progress_pct = min(100, int((upper_sum / 63) * 100))

    turn_footer = "Game Completed (15/15)" if is_game_over else f"Turn {min(state.get('turn_number', 0) + 1, 15)} / 15"

    return f"""
    <div style="background: #111827; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; font-family: system-ui, sans-serif; color: #f8fafc;">
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: rgba(59,130,246,0.1); border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <th style="padding: 10px 12px; text-align: left; color: #60a5fa; font-size: 12px; text-transform: uppercase;">Category</th>
                    <th style="padding: 10px 12px; text-align: center; color: #60a5fa; font-size: 12px; text-transform: uppercase;">Score</th>
                    <th style="padding: 10px 12px; text-align: right; color: #60a5fa; font-size: 12px; text-transform: uppercase;">AI Evaluation</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background: rgba(255,255,255,0.03);"><td colspan="3" style="padding: 6px 12px; font-weight: 700; color: #93c5fd; font-size: 12px;">UPPER SECTION</td></tr>
                {''.join(upper_rows)}
                <tr style="background: rgba(245,158,11,0.08); border-top: 1px solid rgba(245,158,11,0.2); font-weight: 700;">
                    <td style="padding: 8px 12px;">Upper Total (63 for Bonus)</td>
                    <td style="padding: 8px 12px; text-align: center; color: #fbbf24; font-family: monospace;">{upper_sum} / 63</td>
                    <td style="padding: 8px 12px; text-align: right;">
                        <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle;">
                            <div style="width: {progress_pct}%; height: 100%; background: #10b981;"></div>
                        </div>
                    </td>
                </tr>
                <tr style="background: rgba(245,158,11,0.08); border-bottom: 2px solid rgba(255,255,255,0.1); font-weight: 700;">
                    <td style="padding: 8px 12px;">Upper Bonus (+50)</td>
                    <td style="padding: 8px 12px; text-align: center; color: {'#34d399' if has_bonus else '#94a3b8'}; font-family: monospace;">{'50' if has_bonus else '0'}</td>
                    <td style="padding: 8px 12px; text-align: right; font-size: 12px;">{bonus_text}</td>
                </tr>
                <tr style="background: rgba(255,255,255,0.03);"><td colspan="3" style="padding: 6px 12px; font-weight: 700; color: #93c5fd; font-size: 12px;">LOWER SECTION</td></tr>
                {''.join(lower_rows)}
                <tr style="background: rgba(16,185,129,0.15); border-top: 2px solid #10b981; font-weight: 800; font-size: 16px;">
                    <td style="padding: 12px; color: #34d399;">GRAND TOTAL</td>
                    <td style="padding: 12px; text-align: center; color: #34d399; font-family: monospace; font-size: 20px;">{total_score}</td>
                    <td style="padding: 12px; text-align: right; font-size: 13px; color: #a7f3d0;">{turn_footer}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """


def get_ui_bundle():
    """Extract and format complete state bundle for Gradio UI components."""
    st = service.get_game_state()
    sug = service.get_model_suggestions()
    is_game_over = st.get("is_game_over", False)
    total_score = st.get("total_score", 0)

    if is_game_over:
        turn_info = f"🏆 **GAME COMPLETED** | Final Grand Score: **{total_score} points** | All 15 turns finished"
    else:
        current_turn = min(st["turn_number"] + 1, 15)
        turn_info = f"**Turn {current_turn} / 15** | Phase: **{st['current_action_type']}** | Rolls Left: **{st['rolls_remaining']}** | Current Score: **{total_score} pts**"

    dice_html = format_dice_html(st, sug)
    scorecard_html = format_scorecard_html(st, sug)

    if is_game_over:
        bonus_status = "Achieved (+50 pts)" if st.get("has_upper_bonus") else "Missed"
        rating = "🌟 Grandmaster" if total_score >= 260 else "🥇 Master" if total_score >= 220 else "🥈 Expert" if total_score >= 180 else "🥉 Player"
        reasoning_md = f"""
### 🏆 Game Finished!
- **Final Score**: `{total_score} pts` ({rating})
- **Upper Bonus**: `{bonus_status}` (Upper Sum: {st.get('upper_section_sum', 0)}/63)
- Click **"🔄 New Game"** or **"▶ Run Full AI Game"** to play a fresh game!
        """
    else:
        exp_score = f"{round(sug.get('expected_value', 0))} pts" if sug.get("expected_value") else "--"
        reasoning_md = f"""
### 🧠 Model Reasoning & Strategy
- **Expected Final Score ($V(s)$)**: `{exp_score}`
- **Recommendation**: {sug.get('explanation', 'Analyzing hand...')}
        """

    # Build options for category selection dropdown
    cat_options = [
        (f"{cat['display_name']} (Scores +{cat['potential_score']} pts)", cat["index"])
        for cat in st["categories"]
        if not cat["is_filled"]
    ]

    is_dice_phase = st["current_action_type"] == "SELECT_DICE" and not is_game_over
    is_cat_phase = st["current_action_type"] == "SELECT_CATEGORY" and not is_game_over

    return (
        turn_info,
        dice_html,
        scorecard_html,
        reasoning_md,
        gr.update(choices=cat_options, value=cat_options[0][1] if cat_options else None, interactive=is_cat_phase),
        gr.update(interactive=is_dice_phase),
        gr.update(interactive=is_cat_phase),
    )


# ============================================================================
# ZeroGPU Event Handlers
# ============================================================================
@spaces.GPU
def on_roll_dice(selected_keep_positions: List[str]):
    """Execute a dice roll keeping chosen dice."""
    if service.is_game_over:
        gr.Info("Game is complete! Click 'New Game' to start a new match.")
        return get_ui_bundle()

    keep_mask = [False] * 5
    for item in (selected_keep_positions or []):
        try:
            idx = int(item.split()[1]) - 1
            if 0 <= idx < 5:
                keep_mask[idx] = True
        except Exception:
            pass

    service.roll_dice(keep_mask)
    return get_ui_bundle()


@spaces.GPU
def on_apply_ai_keep():
    """Apply the AI's recommended keep selection to the checkboxes."""
    if service.is_game_over:
        return []
    sug = service.get_model_suggestions()
    rec_mask = sug.get("recommended_keep_mask", [False] * 5)
    selected = [f"Die {i+1}" for i, k in enumerate(rec_mask) if k]
    return selected


@spaces.GPU
def on_score_category(category_index: int):
    """Score the selected category and advance to the next turn."""
    if service.is_game_over:
        gr.Info("Game is complete! Click 'New Game' to start a new match.")
        return get_ui_bundle()

    if category_index is not None:
        service.select_category(int(category_index))

    if service.is_game_over:
        gr.Info(f"🎉 Game Completed! Final Score: {service.env.get_score()} points!")

    return get_ui_bundle()


@spaces.GPU
def on_step_ai():
    """Execute 1 optimal RL action."""
    if service.is_game_over:
        service.start_new_game()
        gr.Info("🎲 Started a fresh game!")
        return get_ui_bundle()

    service.step_ai()

    if service.is_game_over:
        gr.Info(f"🎉 Game Completed! Final Score: {service.env.get_score()} points!")

    return get_ui_bundle()


@spaces.GPU
def on_auto_play() -> Generator:
    """Run an entire autonomous game yielding updates turn by turn."""
    # If the game was already over, automatically start a fresh one!
    if service.is_game_over:
        service.start_new_game()
        yield get_ui_bundle()
        time.sleep(0.2)

    step_count = 0
    while not service.is_game_over and step_count < 60:
        service.step_ai()
        step_count += 1
        yield get_ui_bundle()
        time.sleep(0.3)

    if service.is_game_over:
        final_score = int(service.env.get_score())
        gr.Info(f"🎉 AI Game Completed! Final Score: {final_score} points!")


@spaces.GPU
def on_new_game():
    """Reset and start a fresh Yatzy game with random seed."""
    service.start_new_game()
    gr.Info("🎲 New game started! Fresh dice rolled.")
    return get_ui_bundle()


# ============================================================================
# Gradio UI Layout
# ============================================================================
custom_css = """
body, .gradio-container { background-color: #0b0f19 !important; color: #f8fafc !important; }
.btn-primary { background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important; color: white !important; font-weight: bold !important; }
.btn-emerald { background: linear-gradient(135deg, #10b981, #059669) !important; color: white !important; font-weight: bold !important; }
"""

with gr.Blocks(title="Yatzy AI Studio") as demo:
    gr.Markdown(
        """
        # 🎲 Yatzy AI Studio
        ### Deep Reinforcement Learning (PPO) Player & Interactive Assistant
        *Play interactively with real-time Deep RL move recommendations, or watch autonomous agent play.*
        """
    )

    with gr.Row():
        status_banner = gr.Markdown("Loading game...")

    with gr.Row():
        # Left Column: Dice Tray & Controls
        with gr.Column(scale=5):
            dice_display = gr.HTML()

            with gr.Group():
                gr.Markdown("#### 🎯 Dice Hold Selection")
                keep_checkboxes = gr.CheckboxGroup(
                    choices=["Die 1", "Die 2", "Die 3", "Die 4", "Die 5"],
                    label="Select Dice to Keep (Checked = Keep, Unchecked = Reroll)",
                    interactive=True,
                )

            with gr.Row():
                roll_btn = gr.Button("🎲 Roll Dice", elem_classes=["btn-primary"], scale=2)
                ai_keep_btn = gr.Button("✨ Apply AI Keep Selection", scale=2)

            with gr.Group():
                gr.Markdown("#### 📋 Scorecard Selection (Category Phase)")
                cat_dropdown = gr.Dropdown(
                    label="Choose Category to Score",
                    choices=[],
                    interactive=False,
                )
                score_btn = gr.Button("🔒 Score Selected Category", elem_classes=["btn-emerald"])

            reasoning_card = gr.Markdown("Loading AI reasoning...")

            with gr.Accordion("🤖 Autonomous Agent Play", open=True):
                with gr.Row():
                    step_ai_btn = gr.Button("⏩ Step AI Move", scale=1)
                    auto_play_btn = gr.Button("▶ Run Full AI Game", elem_classes=["btn-primary"], scale=1)
                    new_game_btn = gr.Button("🔄 New Game", scale=1)

        # Right Column: Scorecard
        with gr.Column(scale=6):
            scorecard_display = gr.HTML()

    # Event Bindings
    roll_btn.click(
        fn=on_roll_dice,
        inputs=[keep_checkboxes],
        outputs=[status_banner, dice_display, scorecard_display, reasoning_card, cat_dropdown, roll_btn, score_btn],
    )

    ai_keep_btn.click(
        fn=on_apply_ai_keep,
        inputs=[],
        outputs=[keep_checkboxes],
    )

    score_btn.click(
        fn=on_score_category,
        inputs=[cat_dropdown],
        outputs=[status_banner, dice_display, scorecard_display, reasoning_card, cat_dropdown, roll_btn, score_btn],
    )

    step_ai_btn.click(
        fn=on_step_ai,
        inputs=[],
        outputs=[status_banner, dice_display, scorecard_display, reasoning_card, cat_dropdown, roll_btn, score_btn],
    )

    auto_play_btn.click(
        fn=on_auto_play,
        inputs=[],
        outputs=[status_banner, dice_display, scorecard_display, reasoning_card, cat_dropdown, roll_btn, score_btn],
    )

    new_game_btn.click(
        fn=on_new_game,
        inputs=[],
        outputs=[status_banner, dice_display, scorecard_display, reasoning_card, cat_dropdown, roll_btn, score_btn],
    )

    # Initial Load
    demo.load(
        fn=get_ui_bundle,
        inputs=[],
        outputs=[status_banner, dice_display, scorecard_display, reasoning_card, cat_dropdown, roll_btn, score_btn],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css)
