"""Gradio UI."""
from __future__ import annotations

import gradio as gr

from . import config, providers, skills as skills_mod
from .agent import run_turn
from .backend import active_settings, make_backend
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
    async for _ in run_turn(make_backend(settings), mcp, enabled_skills(), active_settings(settings), convo, display):
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
    async for _ in run_turn(make_backend(settings), mcp, enabled_skills(), active_settings(settings), convo, display):
        yield display, convo
    yield display, convo


def active_label() -> str:
    p = settings["provider"]
    return f"**{providers.PROVIDERS[p]['label']}** · `{providers.provider_cfg(settings, p)['model']}`"


async def refresh_models(provider, url, key):
    try:
        ids = await make_backend(settings, provider, {"base_url": url, "api_key": key}).list_models()
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Could not list models: {e}")
        return gr.Dropdown()
    cur = providers.provider_cfg(settings, provider)["model"]
    return gr.Dropdown(choices=ids, value=cur if cur in ids else (ids[0] if ids else None))


async def save_settings_fn(provider, url, key, model, fallback, sys_prompt, temp, top_p, top_k, min_p,
                           rep_pen, max_tokens, max_iter):
    settings.update(provider=provider, claude_fallback=fallback, system_prompt=sys_prompt, temperature=temp,
                    top_p=top_p, top_k=top_k, min_p=min_p, rep_pen=rep_pen, max_tokens=max_tokens,
                    max_tool_iterations=max_iter)
    settings.setdefault("providers", {})[provider] = {"base_url": url, "api_key": key, "model": model}
    config.save_settings(settings)
    try:
        info = await make_backend(settings).ping()
        status = f"✅ Saved. Connected to {providers.PROVIDERS[provider]['label']}: {info}"
    except Exception as e:  # noqa: BLE001
        status = f"Saved, but connection failed: {e}"
    model_dd = gr.Dropdown(choices=[model], value=model)
    return status, active_label(), provider, model_dd


def _persist_model(provider, model):
    settings.setdefault("providers", {}).setdefault(provider, {})["model"] = model
    config.save_settings(settings)


def pick_provider(provider):
    """Chat-tab provider picker: applies immediately; returns updates for the chat and Settings widgets."""
    settings["provider"] = provider
    config.save_settings(settings)
    cfg = providers.provider_cfg(settings, provider)
    saved_key = settings.get("providers", {}).get(provider, {}).get("api_key", "")
    model = gr.Dropdown(choices=[cfg["model"]], value=cfg["model"])
    return model, active_label(), provider, cfg["base_url"], saved_key, model


def pick_model(model):
    if not model:
        return gr.skip(), gr.skip(), gr.skip()
    _persist_model(settings["provider"], model)
    return active_label(), gr.Dropdown(value=model), gr.Dropdown(value=model)


async def refresh_chat_models():
    p = settings["provider"]
    try:
        ids = await make_backend(settings, p).list_models()
    except Exception as e:  # noqa: BLE001
        gr.Warning(f"Could not list models: {e}")
        return gr.skip(), gr.skip()
    cur = providers.provider_cfg(settings, p)["model"]
    dd = gr.Dropdown(choices=ids, value=cur)
    return dd, dd


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
    with gr.Blocks(title="LLM Chat") as demo:
        convo = gr.State([])
        with gr.Row():
            gr.Markdown("### LLM Chat")
            active_md = gr.Markdown(active_label())
            theme_btn = gr.Button("🌓 Theme", size="sm", scale=0, min_width=100, elem_id="theme-btn")
            theme_btn.click(None, js=TOGGLE_JS)
        with gr.Tab("Chat"):
            with gr.Row():
                chat_provider = gr.Dropdown(providers.choices(), value=settings["provider"], label="Provider",
                                            scale=2, interactive=True)
                chat_model = gr.Dropdown([providers.provider_cfg(settings, settings["provider"])["model"]],
                                         value=providers.provider_cfg(settings, settings["provider"])["model"],
                                         allow_custom_value=True, label="Model", scale=4, interactive=True)
                chat_refresh = gr.Button("🔄", scale=0, min_width=50)
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
            cfg = providers.provider_cfg(s, s["provider"])
            provider = gr.Dropdown(providers.choices(), value=s["provider"], label="Provider")
            with gr.Row():
                url = gr.Textbox(cfg["base_url"], label="Base URL", scale=2)
                key = gr.Textbox(s.get("providers", {}).get(s["provider"], {}).get("api_key", ""), type="password",
                                 label="API key (blank = use env var, e.g. OPENAI_API_KEY)", scale=2)
            with gr.Row():
                model = gr.Dropdown([cfg["model"]], value=cfg["model"], allow_custom_value=True,
                                    label="Model (type any id, or refresh the list)", scale=3)
                refresh = gr.Button("🔄 Refresh models", scale=1)
            fallback = gr.Checkbox(s["claude_fallback"], label="Claude: enable server-side refusal fallback (beta)")
            sysp = gr.Textbox(s["system_prompt"], label="System prompt", lines=4)
            gr.Markdown("_Sampling: top-k / min-p / repetition penalty apply to KoboldCpp only; "
                        "Claude ignores all sampling controls._")
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
            refresh.click(refresh_models, [provider, url, key], model)
            save.click(save_settings_fn, [provider, url, key, model, fallback, sysp, temp, top_p, top_k, min_p,
                                          rep, maxt, iters], [status, active_md, chat_provider, chat_model])
            # Chat-tab picker <-> Settings tab stay in sync. (.input fires only on user edits, so
            # programmatic updates from the other tab don't loop.)
            chat_provider.input(pick_provider, chat_provider, [chat_model, active_md, provider, url, key, model])
            chat_model.input(pick_model, chat_model, [active_md, model, chat_model])
            chat_refresh.click(refresh_chat_models, None, [chat_model, model])
            provider.input(pick_provider, provider, [chat_model, active_md, chat_provider, url, key, model])
            model.input(pick_model, model, [active_md, chat_model, model])

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
