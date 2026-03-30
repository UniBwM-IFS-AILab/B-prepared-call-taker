import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from uuid import UUID

    import marimo as mo
    from ems_prepared.util.settings import Settings

    from ems_prepared.policies.pydantic_graph.custom_persistence.resumable_file_persistence import (
        clear_old_run,
    )
    from ems_prepared.policies.pydantic_graph.emergency_main_graph import (
        init_graph,
        run_graph,
    )

    return Settings, UUID, clear_old_run, init_graph, mo, run_graph


@app.cell
def _(mo):
    async def async_out(msg: str):
        return mo.output.append(msg)

    return (async_out,)


@app.cell
async def _(
    Settings,
    UUID,
    async_out,
    clear_old_run,
    init_graph,
    mo,
    run_graph,
):
    deps = Settings(
        name="marimo", user_id=UUID(int=0), session_id=UUID(int=0), emit=async_out
    )
    await clear_old_run(deps.user_id)

    graph, persistence = await init_graph(deps)

    async def answer(answer: str):
        return mo.output.append(
            await run_graph(graph, persistence, deps, answer)
        ).question

    return answer, deps, graph, persistence


@app.cell(disabled=True)
async def _(deps, graph, mo, persistence, run_graph):
    mo.output.append((await run_graph(graph, persistence, deps)).question)
    return


@app.cell(disabled=True)
async def _(answer):
    await answer("Sean")
    return


@app.cell(disabled=True)
async def _(answer):
    await answer("Mainz")
    return


@app.cell
def _(answer, deps, graph, mo, persistence, run_graph):
    async def chat_func(messages, config):
        breakpoint()
        print(len(messages))
        if len(messages) == 0:
            return (await run_graph(graph, persistence, deps)).question
        return await answer(messages[-1].content)

    chat = mo.ui.chat(chat_func)
    seeded, set_seeded = mo.state(False)
    if not seeded:
        chat.value = [
            mo.ai.ChatMessage(
                role="system",
                content="You are a terse helper.",
            ),
            mo.ai.ChatMessage(
                role="assistant",
                content="👋 Hey! Ask me anything.",
            ),
            mo.ai.ChatMessage(role="user", content="What can you do?"),
        ]
        set_seeded(True)
    chat
    return (chat,)


@app.cell
async def _(deps, graph, persistence, run_graph):
    (await run_graph(graph, persistence, deps)).question
    return


@app.cell
async def _(chat, deps, graph, mo, persistence, run_graph):
    chat.value = [
        mo.ai.ChatMessage(
            role="assistant",
            content=(await run_graph(graph, persistence, deps)).question,
        )
    ]
    return


@app.cell
def _(mo):
    def respond(messages: list[mo.ai.ChatMessage], config):
        payload = [{"role": "system", "content": "Be terse. Use markdown."}]
        payload += [
            {"role": chat_message.role, "content": chat_message.content}
            for chat_message in messages
        ]
        return payload  # return text or any renderable object

    chat2 = mo.ui.chat(respond)
    chat2
    return


if __name__ == "__main__":
    app.run()
