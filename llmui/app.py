"""Gradio UI."""
from __future__ import annotations

import gradio as gr

from . import config, skills as skills_mod
from .agent import run_turn
from .backend import Backend
from .mcp_manager import MCPManager

mcp = MCPManager()
settings = config.load_settings()
skills = skills_mod.discover()


def enabled_skills() -> dict:
    off = set(settings["disabled_skills"])
    return {k: v for k, v in skills.items() if k not in off}


async def connect_mcp(text: str | None = None):
    try:
        servers = config.save_mcp_text(text) if text is not None else config.parse_mcp(config.load_mcp_text())
    except ValueError as e:
        return [["-", f"config error: {e}", ""]]
    await mcp.start_all(servers)
    return mcp.status_rows()


async def chat(user_text, display, convo):
    if not user_text.strip():
        yield display, convo, user_text
        return
    display = display + [{"role": "user", "content": user_text}]
    convo = convo + [{"role": "user", "content": user_text}]
    yield display, convo, ""
    async for _ in run_turn(Backend(settings["base_url"]), mcp, enabled_skills(), settings, convo, display):
        yield display, convo, ""
    yield display, convo, ""


async def regenerate(display, convo):
    # Drop everything after the last user message, then rerun.
    while convo and convo[-1]["role"] != "user":
        convo.pop()
    while display and display[-1]["role"] != "user":
        display.pop()
    if not convo:
        yield display, convo
        return
    async for _ in run_turn(Backend(settings["base_url"]), mcp, enabled_skills(), settings, convo, display):
        yield display, convo
    yield display, convo


async def save_settings_fn(url, model, sys_prompt, temp, top_p, top_k, min_p, rep_pen, max_tokens, max_iter):
    settings.update(base_url=url, model=model, system_prompt=sys_prompt, temperature=temp, top_p=top_p,
                    top_k=top_k, min_p=min_p, rep_pen=rep_pen, max_tokens=max_tokens, max_tool_iterations=max_iter)
    config.save_settings(settings)
    try:
        return f"✅ Saved. Connected to model: {await Backend(url).ping()}"
    except Exception as e:  # noqa: BLE001
        return f"Saved, but connection failed: {e}"


def skill_rows():
    return [[k, v.description, k not in settings["disabled_skills"]] for k, v in skills.items()]


def rescan():
    global skills
    skills = skills_mod.discover()
    return skill_rows()


def toggle_skills(rows):
    df = rows.values.tolist() if hasattr(rows, "values") else rows
    settings["disabled_skills"] = [r[0] for r in df if not r[2]]
    config.save_settings(settings)
    return "Saved."


def preview_skill(evt: gr.SelectData):
    name = evt.row_value[0]
    s = skills.get(name)
    return s.body if s else ""


# Dark by default; choice persisted in localStorage ("llmui-theme").
THEME_JS = """
() => {
  const apply = (t) => document.body.classList.toggle('dark', t === 'dark');
  apply(localStorage.getItem('llmui-theme') || 'dark');
}
"""
TOGGLE_JS = """
() => {
  const dark = !document.body.classList.contains('dark');
  document.body.classList.toggle('dark', dark);
  localStorage.setItem('llmui-theme', dark ? 'dark' : 'light');
}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="KoboldCpp Chat") as demo:
        convo = gr.State([])
        with gr.Row():
            gr.Markdown("### KoboldCpp Chat")
            theme_btn = gr.Button("🌓 Theme", size="sm", scale=0, min_width=100, elem_id="theme-btn")
            theme_btn.click(None, js=TOGGLE_JS)
        with gr.Tab("Chat"):
            bot = gr.Chatbot(height=560, placeholder="Say something…")
            box = gr.Textbox(placeholder="Message (Enter to send)", show_label=False, autofocus=True)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                regen = gr.Button("Regenerate")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
            sub = box.submit(chat, [box, bot, convo], [bot, convo, box])
            clk = send.click(chat, [box, bot, convo], [bot, convo, box])
            rg = regen.click(regenerate, [bot, convo], [bot, convo])
            stop.click(None, cancels=[sub, clk, rg])
            clear.click(lambda: ([], []), None, [bot, convo], cancels=[sub, clk, rg])

        with gr.Tab("Settings"):
            s = settings
            url = gr.Textbox(s["base_url"], label="KoboldCpp URL")
            model = gr.Textbox(s["model"], label="Model name (sent to API)")
            sysp = gr.Textbox(s["system_prompt"], label="System prompt", lines=4)
            with gr.Row():
                temp = gr.Slider(0, 2, s["temperature"], step=0.05, label="Temperature")
                top_p = gr.Slider(0, 1, s["top_p"], step=0.01, label="Top P")
                min_p = gr.Slider(0, 1, s["min_p"], step=0.01, label="Min P")
            with gr.Row():
                top_k = gr.Slider(0, 200, s["top_k"], step=1, label="Top K")
                rep = gr.Slider(1, 2, s["rep_pen"], step=0.01, label="Repetition penalty")
                maxt = gr.Slider(16, 8192, s["max_tokens"], step=16, label="Max tokens")
                iters = gr.Slider(1, 30, s["max_tool_iterations"], step=1, label="Max tool iterations")
            save = gr.Button("Save & test connection", variant="primary")
            status = gr.Markdown()
            save.click(save_settings_fn, [url, model, sysp, temp, top_p, top_k, min_p, rep, maxt, iters], status)

        with gr.Tab("MCP"):
            gr.Markdown('Claude-Desktop-style config. stdio: `{"command","args","env"}`; remote: `{"url","headers"}`.')
            mcp_text = gr.Code(config.load_mcp_text(), language="json", label="mcp.json")
            reconnect = gr.Button("Save & reconnect", variant="primary")
            table = gr.Dataframe(headers=["Server", "Status", "Tools"], value=mcp.status_rows(),
                                 interactive=False, wrap=True)
            reconnect.click(connect_mcp, mcp_text, table)

        with gr.Tab("Skills"):
            gr.Markdown("Skills live in `./skills/<name>/SKILL.md` or `~/.config/llmui/skills/`. Untick to disable.")
            sk_table = gr.Dataframe(headers=["Name", "Description", "Enabled"], datatype=["str", "str", "bool"],
                                    value=skill_rows(), interactive=True, static_columns=[0, 1])
            with gr.Row():
                rescan_btn = gr.Button("Rescan")
                apply_btn = gr.Button("Apply enabled/disabled", variant="primary")
            sk_status = gr.Markdown()
            sk_preview = gr.Textbox(label="SKILL.md preview (click a row)", lines=10, interactive=False)
            rescan_btn.click(rescan, None, sk_table)
            apply_btn.click(toggle_skills, sk_table, sk_status)
            sk_table.select(preview_skill, None, sk_preview)

        demo.load(connect_mcp, None, table)
        demo.load(None, js=THEME_JS)
    return demo
