import json
import logging
import re
from time import perf_counter
from typing import List

from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config
from app.models.llm_provider import DEFAULT_LLM_PROVIDER_ID, get_llm_provider

_max_retries = 5
MIN_SCRIPT_PARAGRAPH_NUMBER = 1
MAX_SCRIPT_PARAGRAPH_NUMBER = 10
MAX_SCRIPT_PROMPT_LENGTH = 2000
MAX_SCRIPT_SYSTEM_PROMPT_LENGTH = 8000
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*$", re.IGNORECASE | re.DOTALL)
_URL_USERINFO_RE = re.compile(
    r"((?:https?|wss?)://)([^/\s?#@]*:[^/\s?#@]*@)", re.IGNORECASE
)
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|token|key|secret|password)=)([^&#\s]+)",
    re.IGNORECASE,
)

DEFAULT_SCRIPT_SYSTEM_PROMPT = """
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.
""".strip()


def _normalize_text_response(content, llm_provider: str) -> str:
    # Разные SDK для LLM при ошибке или срабатывании фильтра могут вернуть None,
    # пустую строку и даже нестроковый объект. Делаем единую подстраховку, чтобы
    # последующий вызов `.replace()` не бросил что-нибудь вроде ошибки атрибута у `NoneType`.
    if content is None:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    if not isinstance(content, str):
        raise TypeError(
            f"[{llm_provider}] returned non-text content: {type(content).__name__}"
        )

    # Reasoning-модели вроде MiniMax M3 и DeepSeek R1 могут возвращать внутренние
    # рассуждения в обёртке `<think>...</think>`. Сценарию видео и ключевым словам
    # нужен только итоговый произносимый текст: без единой очистки в сервисном слое WebUI, субтитры и озвучка приняли бы ход рассуждений за основной текст.
    content = _THINK_BLOCK_RE.sub("", content)
    content = _UNCLOSED_THINK_BLOCK_RE.sub("", content).strip()
    if not content:
        raise ValueError(f"[{llm_provider}] returned empty text content")

    # ``strip()`` выше уже убрал пробелы по краям. Здесь обязательно сохранить
    # одинарные и двойные переводы строк внутри текста: генерация сценария разделяет абзацы двойным переводом, а обработка субтитров читает пользовательский текст построчно.
    return content


def _sanitize_error_message(error: object) -> str:
    """
    Очищает сообщение об ошибке, возвращаемое в WebUI и API, чтобы не утекли
    учётные данные из пользовательского base_url.

    Некоторые OpenAI-совместимые SDK вставляют URL запроса в текст исключения как
    есть. Если пользователь настроил прокси-шлюз как
    `https://user:pass@example.com/v1`, то прямой возврат `str(e)` раскрыл бы
    пароль странице, вызывающей стороне API и последующим логам. Здесь
    обрабатывается только текст ошибки — реальный адрес запроса не меняется,
    чтобы не задеть рабочую цепочку вызовов.
    """
    message = str(error)
    message = _URL_USERINFO_RE.sub(r"\1***:***@", message)
    message = _SENSITIVE_QUERY_RE.sub(r"\1***", message)
    return message


def _extract_chat_completion_text(response, llm_provider: str) -> str:
    # OpenAI-совместимый эндпоинт при сбое может вернуть объект ответа без choices
    # либо с пустыми choices, message или content.
    # Проверяем структуру единообразно, чтобы не получить низкоуровневую ошибку
    # доступа к атрибуту вроде `NoneType is not subscriptable`.
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError(f"[{llm_provider}] returned empty choices")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"[{llm_provider}] returned empty message")

    content = getattr(message, "content", None)
    return _normalize_text_response(content, llm_provider)


def _get_response_field(value, key: str):
    """Чтение поля, работающее и со словарём, и с объектом ответа SDK."""
    if isinstance(value, dict):
        return value.get(key)

    try:
        return value[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(value, key, None)


def _extract_qwen_generation_text(response) -> str:
    """
    Извлекает текст из ответа DashScope Generation.

    При вызове через `messages` Qwen возвращает chat-структуру:
    `output.choices[0].message.content`; `output.text` приходит только в старом
    completion-формате. Поддерживаем оба пути, чтобы при `output.text` равном
    None последующий `.replace()` не вызывал недиагностируемый AttributeError.
    """
    output = _get_response_field(response, "output")
    choices = _get_response_field(output, "choices") if output else None
    if choices is not None:
        if not choices:
            logger.warning("Qwen returned an empty choices list")
            raise ValueError("[qwen] returned empty choices")

        first_choice = choices[0]
        message = _get_response_field(first_choice, "message")
        content = _get_response_field(message, "content") if message else None
        if content is not None:
            return _normalize_text_response(content, "qwen")

    text = _get_response_field(output, "text") if output else None
    return _normalize_text_response(text, "qwen")


def _generate_response(prompt: str, app_config=None) -> str:
    try:
        # Во время генерации видео WebUI позволяет пользователю готовить следующий
        # текст. Вызывающая сторона может передать снимок конфигурации на момент
        # отправки: тогда во время повторов запроса к модели завершившаяся фоновая задача, применив новую конфигурацию, не переключит его на другого провайдера, Base URL или модель.
        runtime_app_config = app_config if app_config is not None else config.app
        llm_provider = str(
            runtime_app_config.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
        ).lower()
        provider = get_llm_provider(llm_provider)
        if provider is None:
            raise ValueError(f"{llm_provider}: unsupported llm provider")

        logger.info(f"llm provider: {llm_provider}")
        api_key = runtime_app_config.get(provider.config_key("api_key"), "")
        configured_model = runtime_app_config.get(provider.config_key("model_name"), "")
        model_name = provider.resolve_model_name(configured_model)
        if configured_model and model_name != configured_model:
            logger.warning(
                f"{llm_provider} model '{configured_model}' is deprecated, "
                f"fallback to '{model_name}'"
            )
        configured_base_url = runtime_app_config.get(
            provider.config_key("base_url"), ""
        )
        base_url = provider.resolve_base_url(configured_base_url)
        if configured_base_url and configured_base_url.strip().rstrip("/") in {
            url.rstrip("/") for url in provider.deprecated_base_urls
        }:
            logger.warning(
                f"{llm_provider} base URL '{configured_base_url}' is deprecated, "
                f"fallback to '{base_url}'"
            )
        adapter = provider.adapter
        api_version = ""

        # Адрес Ollama по умолчанию зависит от того, работаем ли мы в контейнере, и
        # статическим значением Registry быть не может. Registry по-прежнему отвечает за модели и обязательные поля, а различия окружения разрешаются здесь.
        if llm_provider == "ollama":
            api_key = "ollama"
            if not base_url:
                base_url = config.get_default_ollama_base_url()

        if adapter == "azure":
            api_version = runtime_app_config.get(
                provider.config_key("api_version"), "2024-02-15-preview"
            )

        extra_values = {
            field.config_suffix: (
                runtime_app_config.get(provider.config_key(field.config_suffix), "")
                or field.default_value
            )
            for field in provider.extra_fields
        }

        if provider.requires_api_key and not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if provider.requires_model_name and not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if provider.requires_base_url and not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )

        for field in provider.extra_fields:
            if field.required and not extra_values[field.config_suffix]:
                raise ValueError(
                    f"{llm_provider}: {field.config_suffix} is not set, "
                    "please set it in the config.toml file."
                )

        if adapter == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            dashscope.api_key = api_key
            response = dashscope.Generation.call(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, GenerationResponse):
                    status_code = response.status_code
                    if status_code != 200:
                        raise Exception(
                            f'[{llm_provider}] returned an error response: "{response}"'
                        )

                    return _extract_qwen_generation_text(response)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}"'
                    )
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        if adapter == "gemini":
            from google import genai
            from google.genai import types

            http_options = types.HttpOptions(base_url=base_url) if base_url else None
            generation_config = types.GenerateContentConfig(
                temperature=0.5,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_ONLY_HIGH",
                    ),
                ],
            )

            try:
                # Новая версия google-genai отдаёт сервисы моделей через единый Client.
                # Контекстный менеджер закрывает нижележащее HTTP-соединение по окончании запроса, чтобы при частой генерации не копились ресурсы соединений.
                with genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                ) as client:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=generation_config,
                    )
                generated_text = response.text
            except (AttributeError, IndexError, ValueError) as e:
                logger.warning(f"gemini returned invalid response content: {str(e)}")
                raise ValueError(f"[{llm_provider}] returned invalid response content")

            return _normalize_text_response(generated_text, llm_provider)

        if adapter == "cloudflare_ai_gateway":
            account_id = extra_values["account_id"]
            gateway_id = extra_values["gateway_id"]
            # Рекомендуемый сейчас Cloudflare REST API для AI Gateway совместим с OpenAI SDK.
            # Account ID нужен для сборки единого эндпоинта, Gateway ID выбирается
            # заголовком запроса; специальный эндпоинт Workers AI /ai/run/{model} больше не вызывается.
            client = OpenAI(
                api_key=api_key,
                base_url=(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
                ),
                default_headers={"cf-aig-gateway-id": gateway_id},
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "litellm":
            import litellm

            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )

            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                drop_params=True,
            )

            if not response:
                raise ValueError(f"[{llm_provider}] returned empty response")
            if not getattr(response, "choices", None):
                raise ValueError(f"[{llm_provider}] returned empty response")

            return _extract_chat_completion_text(response, llm_provider)

        if adapter == "azure":
            # SDK Azure OpenAI формирует специальный адрес запроса из `azure_endpoint` и
            # `api_version`, поэтому переиспользовать инициализацию через `base_url` для
            # обычных OpenAI-совместимых провайдеров ниже нельзя. Запрос выполняется и
            # сразу возвращается прямо в ветке Azure, чтобы клиента не перекрыл последующий fallback: иначе настроенные пользователем учётные данные Azure прошли бы проверку, но в реальном запросе не использовались.
            logger.info(f"requesting azure chat completion, model: {model_name}")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=base_url,
            )
            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    return _extract_chat_completion_text(response, llm_provider)
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        if adapter == "modelscope":
            content = ""
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": False},
                stream=True,
            )
            if response:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content += delta.content

                if not content.strip():
                    raise ValueError("Empty content in stream response")

                return _normalize_text_response(content, llm_provider)
            else:
                raise Exception(f"[{llm_provider}] returned an empty response")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        if response:
            if isinstance(response, ChatCompletion):
                return _extract_chat_completion_text(response, llm_provider)
            else:
                raise Exception(
                    f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                    f"connection and try again."
                )
        else:
            raise Exception(
                f"[{llm_provider}] returned an empty response, please check your network connection and try again."
            )

    except Exception as e:
        return f"Error: {_sanitize_error_message(e)}"


def test_connection() -> tuple[bool, str, float]:
    """
    Отправляет один минимальный запрос с текущими настройками провайдера, чтобы
    проверить работоспособность реальной цепочки генерации.

    Тест подключения напрямую переиспользует `_generate_response()`, поэтому
    покрывает API-ключ, Base URL, имя модели и специфичные для провайдера поля,
    но не заходит в логику повторов генерации сценария и не отправляет тему или
    текст пользователя. Возвращает статус успеха, сообщение об ошибке и
    длительность запроса.
    """
    started_at = perf_counter()
    response = _generate_response(prompt="Reply with exactly: OK")
    elapsed = perf_counter() - started_at

    if not response:
        error_message = "LLM returned an empty response"
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    if response.startswith("Error:"):
        error_message = response.removeprefix("Error:").strip()
        logger.warning(f"llm connection test failed: {error_message}")
        return False, error_message, elapsed

    logger.info(f"llm connection test succeeded, elapsed: {elapsed:.2f}s")
    return True, "", elapsed


def _limit_script_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # Слой API уже проверяет длину через Pydantic. Подстраховка здесь нужна, чтобы
    # при прямом вызове generate_script из WebUI или внутренних сервисов слишком
    # длинный промпт не ушёл в модель и не привёл к аномальным тратам токенов и сбою запроса.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _normalize_script_paragraph_number(paragraph_number: int | None) -> int:
    try:
        value = int(paragraph_number or MIN_SCRIPT_PARAGRAPH_NUMBER)
    except (TypeError, ValueError):
        value = MIN_SCRIPT_PARAGRAPH_NUMBER

    if value < MIN_SCRIPT_PARAGRAPH_NUMBER or value > MAX_SCRIPT_PARAGRAPH_NUMBER:
        # WebUI и API оба ограничивают диапазон. Здесь подстраховка для внутренних
        # вызовов, чтобы некорректный параметр не раздул стоимость генерации в LLM и не дал пустой результат.
        logger.warning(
            f"script paragraph_number is out of range and will be clamped: {value}"
        )
        return max(MIN_SCRIPT_PARAGRAPH_NUMBER, min(value, MAX_SCRIPT_PARAGRAPH_NUMBER))

    return value


def build_script_prompt(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )

    # Разделяем склейку «правил генерации сценария» и «контекста выполнения». Тогда
    # даже опытный пользователь, переопределивший системный промпт, не потеряет тему видео, язык и число абзацев — параметры, обязательные для каждой генерации.
    prompt = custom_system_prompt or DEFAULT_SCRIPT_SYSTEM_PROMPT
    prompt += f"""

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".rstrip()
    if language:
        prompt += f"\n- language: {language}"
    if video_script_prompt:
        prompt += f"""

# Additional User Requirements:
{video_script_prompt}
""".rstrip()

    return prompt


def generate_script(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    app_config=None,
) -> str:
    paragraph_number = _normalize_script_paragraph_number(paragraph_number)
    video_script_prompt = _limit_script_text(
        video_script_prompt, MAX_SCRIPT_PROMPT_LENGTH, "video_script_prompt"
    )
    custom_system_prompt = _limit_script_text(
        custom_system_prompt, MAX_SCRIPT_SYSTEM_PROMPT_LENGTH, "custom_system_prompt"
    )
    prompt = build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    final_script = ""
    logger.info(
        "generating video script: "
        f"subject={video_subject}, paragraph_number={paragraph_number}, "
        f"has_custom_prompt={bool(video_script_prompt.strip())}, "
        f"has_custom_system_prompt={bool(custom_system_prompt.strip())}"
    )

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax.  Use non-greedy .*? so each bracket/paren
        # group is removed independently; the greedy form would eat all text
        # between the first opener and the last closer on the same line.
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt=prompt)
            else:
                response = _generate_response(prompt=prompt, app_config=app_config)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # Some upstream providers may return quota errors as plain text.
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries - 1:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def _strip_code_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from an LLM response.

    Non-OpenAI providers (Claude, Gemini, …) frequently wrap JSON output in a
    ```json … ``` fence even when asked to return raw JSON. Removing it lets the
    first json.loads() succeed instead of falling through to the regex recovery
    path (and spuriously logging a warning). Mirrors the DOTALL handling already
    used in _parse_social_metadata().
    """
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def generate_terms(
    video_subject: str,
    video_script: str,
    amount: int = 5,
    match_script_order: bool = False,
    app_config=None,
) -> List[str]:
    if match_script_order:
        goal = (
            f"Generate {amount} chronological stock-video search terms that follow "
            "the order of topics in the video script."
        )
        ordering_rule = (
            "6. keep the terms in the same order as the script narration; "
            "earlier terms must describe earlier visual moments."
        )
        # В режиме упорядоченных ключевых слов число примеров должно совпадать с amount:
        # иначе фиксированные 4 примера собьют модель, и для длинного текста она вернёт слишком мало ключевых слов, ухудшив покрытие материалами.
        example_terms = [
            "opening visual topic",
            *[f"script visual topic {index}" for index in range(2, max(amount, 1))],
            "final visual topic",
        ]
        output_example = json.dumps(example_terms[:amount], ensure_ascii=False)
    else:
        goal = (
            f"Generate {amount} search terms for stock videos, depending on the "
            "subject of a video."
        )
        ordering_rule = ""
        output_example = (
            '["search term 1", "search term 2", "search term 3",'
            '"search term 4", "search term 5"]'
        )

    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
{goal}

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.
{ordering_rule}

## Output Example:
{output_example}

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}, match_script_order: {match_script_order}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            if app_config is None:
                response = _generate_response(prompt)
            else:
                response = _generate_response(prompt, app_config=app_config)
            if response.startswith("Error: "):
                # Публичный тип возврата generate_terms — List[str]. Если вернуть текст ошибки
                # провайдера как есть, потребитель, проверяющий лишь пустоту, примет непустую
                # строку за успех, а цикл загрузки материалов ещё и пройдётся по тексту ошибки
                # посимвольно, породив бессмысленные внешние запросы. Возвращаем пустой список,
                # чтобы слой оркестрации задач завершил задачу прямо в месте реального сбоя.
                logger.error(f"failed to generate video terms: {response}")
                return []
            search_terms = json.loads(_strip_code_fence(response))
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response, re.DOTALL)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        # Оставляем процедуру повтора, но обязательно записываем нестандартный JSON,
                        # который вернула LLM: иначе при пустых поисковых словах будет не понять,
                        # проблема в формате модели или в логике разбора.
                        logger.warning(f"failed to generate video terms: {str(e)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries - 1:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


# =============================================================================
# Social publishing metadata
#
# Генерирует title, caption и hashtags, типичные для публикации в сервисах
# коротких видео, по теме и сценарию ролика. Возможность переиспользует только
# существующего LLM-провайдера, не подключает внешние сервисы публикации и не влияет на основную цепочку генерации видео.
# =============================================================================

# У разных площадок свои предпочтения по длине текста и числу хэштегов.
# Берём консервативные верхние границы, чтобы вызывающей стороне не пришлось повторно обрезать слишком длинный ответ модели.
SOCIAL_PLATFORMS = {
    "tiktok": {"title_max": 100, "caption_max": 2200, "hashtag_count": 5},
    "youtube_shorts": {"title_max": 100, "caption_max": 5000, "hashtag_count": 3},
    "instagram_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 8},
    "facebook_reels": {"title_max": 125, "caption_max": 2200, "hashtag_count": 5},
}
DEFAULT_SOCIAL_PLATFORM = "tiktok"
DEFAULT_SOCIAL_LANGUAGE = "auto"
MAX_SOCIAL_SUBJECT_LENGTH = 500
MAX_SOCIAL_SCRIPT_LENGTH = 8000
MAX_SOCIAL_LANGUAGE_LENGTH = 64

SOCIAL_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "youtube_shorts": "YouTube Shorts",
    "instagram_reels": "Instagram Reels",
    "facebook_reels": "Facebook Reels",
}

# Универсальные запасные теги на случай недоступности LLM. Намеренно не привязаны
# к конкретной стране или языку, чтобы API возвращал пригодную структуру и для китайских, и для английских, и для вьетнамских сценариев.
DEFAULT_SOCIAL_HASHTAGS = [
    "#shorts",
    "#viral",
    "#trending",
    "#fyp",
    "#video",
    "#reels",
    "#creator",
    "#content",
]


def _resolve_social_platform(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    return value if value in SOCIAL_PLATFORMS else DEFAULT_SOCIAL_PLATFORM


def _normalize_social_language(language: str | None) -> str:
    value = (language or DEFAULT_SOCIAL_LANGUAGE).strip()
    if len(value) > MAX_SOCIAL_LANGUAGE_LENGTH:
        logger.warning(
            "social metadata language is too long and will be truncated to "
            f"{MAX_SOCIAL_LANGUAGE_LENGTH} characters."
        )
        value = value[:MAX_SOCIAL_LANGUAGE_LENGTH]
    return value or DEFAULT_SOCIAL_LANGUAGE


def _limit_social_text(text: str | None, max_length: int, field_name: str) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value

    # Слой API ограничивает длину. Подстраховка здесь нужна, чтобы при внутренних
    # вызовах или будущем прямом вызове из WebUI слишком длинный текст не ушёл в модель и не вызвал аномальных трат токенов.
    logger.warning(
        f"{field_name} is too long and will be truncated to {max_length} characters."
    )
    return value[:max_length]


def _social_language_instruction(language: str | None) -> str:
    language = _normalize_social_language(language)
    if language.lower() == DEFAULT_SOCIAL_LANGUAGE:
        return (
            "Use the same language as the video subject and script. If the subject "
            "and script use different languages, prefer the script language."
        )

    return f'Write "title" and "caption" in this language: {language}.'


def _clamp_text(text, max_length: int) -> str:
    value = ("" if text is None else str(text)).strip()
    if max_length and len(value) > max_length:
        return value[:max_length].rstrip()
    return value


def _normalize_hashtags(raw, count: int) -> List[str]:
    """
    Приводит хэштеги, возвращённые LLM, к единому формату `#tag`.

    LLM может вернуть строку, массив, словосочетание с пробелами, повторяющиеся
    теги или содержимое со знаками препинания. Централизованная очистка даёт
    стабильную структуру ответа эндпоинта и не даёт при публикации появиться
    пустым, повторяющимся или нестандартным по формату хэштегам.
    """
    if isinstance(raw, str):
        candidates = re.split(r"[\s,]+", raw)
    elif isinstance(raw, (list, tuple)):
        # Каждый элемент массива считается цельным тегом, поэтому "du lich"
        # превращается в "#dulich", а не в два отдельных тега.
        candidates = [str(entry) for entry in raw]
    else:
        candidates = []

    seen = set()
    result: List[str] = []
    for item in candidates:
        tag = re.sub(r"[^\w]", "", item, flags=re.UNICODE)
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(f"#{tag}")
        if count and len(result) >= count:
            break
    return result


def build_social_metadata_prompt(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> str:
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    platform = _resolve_social_platform(platform)
    spec = SOCIAL_PLATFORMS[platform]
    label = SOCIAL_PLATFORM_LABELS.get(platform, platform)
    language_instruction = _social_language_instruction(language)

    prompt = f"""
# Role: Short-Video Social Media Copywriter

## Goal
Write engaging publishing metadata for a short video that will be posted on {label}.

## Constraints
1. Respond ONLY with a single valid minified JSON object. No markdown, no code fences, no commentary.
2. The JSON must contain exactly these keys: "title", "caption", "hashtags".
3. "title": a catchy hook, at most {spec["title_max"]} characters.
4. "caption": an engaging description that ends with a call to action, at most {spec["caption_max"]} characters. Do not put hashtags inside the caption.
5. "hashtags": a JSON array of exactly {spec["hashtag_count"]} strings. Each must start with "#", contain no spaces, and be relevant to the topic and to {label}.
6. {language_instruction}

## Output Example
{{"title":"...","caption":"...","hashtags":["#example","#video"]}}

## Context
### Video Subject
{video_subject}

### Video Script
{video_script}
""".strip()
    return prompt


def _parse_social_metadata(response: str, platform: str) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]

    data = None
    try:
        data = json.loads(_strip_code_fence(response))
    except Exception:
        # Часть моделей оборачивает JSON пояснительным текстом или markdown-заграждением.
        # Вызывающей стороне API нужна только стабильная структура, поэтому пробуем извлечь первый JSON-объект.
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if match:
            data = json.loads(match.group())

    if not isinstance(data, dict):
        raise ValueError("social metadata response is not a JSON object")

    title = _clamp_text(data.get("title", ""), spec["title_max"])
    caption = _clamp_text(data.get("caption", ""), spec["caption_max"])
    hashtags = _normalize_hashtags(data.get("hashtags", []), spec["hashtag_count"])

    if not title and not caption:
        raise ValueError("social metadata response is missing both title and caption")

    return {"title": title, "caption": caption, "hashtags": hashtags}


def _fallback_social_metadata(
    video_subject: str, video_script: str, platform: str
) -> dict:
    spec = SOCIAL_PLATFORMS[_resolve_social_platform(platform)]
    subject = (video_subject or "").strip()
    script = (video_script or "").strip()

    title = subject
    if not title and script:
        # Если темы нет, собираем title из первой фразы сценария, чтобы эндпоинт не вернул пустой заголовок.
        title = re.split(r"(?<=[.!?。！？])\s+", script)[0]

    return {
        "title": _clamp_text(title, spec["title_max"]),
        "caption": _clamp_text(script or subject, spec["caption_max"]),
        "hashtags": _normalize_hashtags(DEFAULT_SOCIAL_HASHTAGS, spec["hashtag_count"]),
    }


def generate_social_metadata(
    video_subject: str,
    video_script: str = "",
    language: str = DEFAULT_SOCIAL_LANGUAGE,
    platform: str = DEFAULT_SOCIAL_PLATFORM,
) -> dict:
    """
    Генерирует метаданные текста для публикации короткого видео.

    Структура возврата фиксирована: `{"title": str, "caption": str, "hashtags": List[str]}`.
    Если LLM недоступна или вернула некорректный формат, результат деградирует до
    универсальной эвристики: вызывающая сторона API всегда получает структуру,
    пригодную для показа и правки перед публикацией.
    """
    platform = _resolve_social_platform(platform)
    language = _normalize_social_language(language)
    video_subject = _limit_social_text(
        video_subject, MAX_SOCIAL_SUBJECT_LENGTH, "video_subject"
    )
    video_script = _limit_social_text(
        video_script, MAX_SOCIAL_SCRIPT_LENGTH, "video_script"
    )
    prompt = build_social_metadata_prompt(
        video_subject=video_subject,
        video_script=video_script,
        language=language,
        platform=platform,
    )
    logger.info(f"generating social metadata: platform={platform}, language={language}")

    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if isinstance(response, str) and "Error: " in response:
                logger.error(f"failed to generate social metadata: {response}")
                break
            metadata = _parse_social_metadata(response, platform)
            logger.success(f"completed: \n{metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"failed to parse social metadata: {str(e)}")

        if i < _max_retries - 1:
            logger.warning(
                f"failed to generate social metadata, trying again... {i + 1}"
            )

    logger.warning("falling back to heuristic social metadata")
    return _fallback_social_metadata(video_subject, video_script, platform)


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
