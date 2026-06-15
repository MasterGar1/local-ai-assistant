import ollama
from math import floor

def download(model: str) -> None:
    status = ollama.pull(model, stream=True)
    for chunk in status:
        if chunk.completed is None or chunk.total is None:
            continue
        print(floor(float(chunk.completed / chunk.total) * 100), '%')

class Model:
    def __init__(self, mn: str, sp: str | None = None):
        self.model_name: str = mn
        self.system_prompt: dict[str, str] | None = None
        if not self._is_loaded():
            self._load()
        self.history: list[dict[str, str]] = []
        if sp is not None:
            self.set_system_prompt(sp)

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = {'role': 'system', 'content': system_prompt}

    def _is_loaded(self) -> bool:
        return any(map(lambda m: m.model == self.model_name, ollama.ps().models))

    def _load(self) -> None:
        print(f'[System] Loading: {self.model_name}')
        try:
            ollama.chat(self.model_name, messages=[])
            print(f'[System] Loaded: {self.model_name}')
        except ollama.ResponseError as e:
            if e.status_code == 404:
                ollama.pull(model=self.model_name)
                print(f'[System] Pulled: {self.model_name}')
                self._load()

    def _unload(self) -> None:
        ollama.chat(self.model_name, messages=[], keep_alive=0)
        print(f'[System] Unloaded: {self.model_name}')

    async def generate_response(self, prompt: str):
      message: dict[str, str] = {'role': 'user', 'content': prompt}
      self.history.append(message)

      messages: list[dict[str, str]] = ([self.system_prompt] + self.history) if self.system_prompt is not None else self.history

      print(f'{self.model_name}: ', end='')
      async for part in await ollama.AsyncClient().chat(model=self.model_name, messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)
      print()
