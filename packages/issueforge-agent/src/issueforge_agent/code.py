from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```(?:python|cuda|cpp)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    matches = _CODE_FENCE.findall(text or "")
    if matches:
        # 优先包含 ModelNew 的代码块
        for block in matches:
            if "class ModelNew" in block:
                return block.strip()
        return max(matches, key=len).strip()
    return (text or "").strip()
