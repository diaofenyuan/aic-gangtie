from __future__ import annotations

from io import StringIO

from gas_power import runtime


def test_progress_bar_uses_one_fixed_physical_line(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    def fake_tqdm(iterable=None, **kwargs):
        created.append(dict(kwargs))
        return object()

    monkeypatch.setattr("tqdm.auto.tqdm", fake_tqdm)
    first = runtime.progress_bar(total=10, desc="数据准备", leave=True)
    second = runtime.progress_bar(total=20, desc="滚动验证", leave=True)

    assert first is not second
    assert [item["desc"] for item in created] == ["数据准备", "滚动验证"]
    assert all(item["position"] == 0 for item in created)
    assert all(item["mininterval"] == 0.2 for item in created)


def test_progress_status_updates_in_place() -> None:
    output = StringIO()

    with runtime.progress_bar(
        total=3,
        desc="数据准备",
        leave=True,
        file=output,
        mininterval=0.0,
    ) as progress:
        for index in range(3):
            progress.set_postfix_str(f"项目 {index + 1}", refresh=True)
            progress.update(1)

    rendered = output.getvalue()
    assert rendered.count("\n") == 1
    assert rendered.count("\r") >= 3
