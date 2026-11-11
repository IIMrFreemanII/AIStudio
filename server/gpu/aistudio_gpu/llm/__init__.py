"""LLMs on the GPU server: any Hugging Face model by link, served by vLLM behind the front desk's OpenAI-compatible
routes (``/v1/llm/…``, see ``routes.py``).

- ``facts.py``: what a link points at (architecture, weight bytes, quantization, context, vision, chat template).
- ``sizing.py``: which GPUs it fits on, and the cheapest of them.
- ``profiles.py``: tool-call and reasoning parsers, and engine flags, per model.
- ``store.py``: installed models (``<models>/llm/<id>/llm.json`` + ``weights/``).
- ``engine.py``: the vLLM process (on Modal and on a GPU box), and the fake engine for tests.
- ``proxy.py``: relays a chat stream, keeps it alive during a cold start, adds usage and cost.
- ``service.py``: the front desk's view (``LocalLLMService``; Modal's is in ``modal_app.py``).
"""
