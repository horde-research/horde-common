import json
import re
from typing import Any, Dict, List, Union

import openai


_TEMPLATE_API = """
Сіз қазақ тілінде жауап беретін білімді, пайдалы көмекшісіз. Сұрақты және жауап нұсқаларын мұқият оқып,
ең дұрысын бір ғана әріппен (A, B, C, D) белгілеңіз.  
Жауапты **тек** төмендегі JSON құрылымында қайтарыңыз:

```json
{{"answer": "A"}}
```

Cұрақ: {prompt}
A) {a}
B) {b}
C) {c}
D) {d}

Жауап:
"""


def inference_open_ai(
    prompt: str,
) -> Union[Dict[str, Any], str]:
    messages = [
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )

    assistant_reply = response.choices[0].message.content.strip()

    json_match = re.search(r"\{.*\}", assistant_reply, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass 

    return assistant_reply
