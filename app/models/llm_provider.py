from dataclasses import dataclass


DEFAULT_LLM_PROVIDER_ID = "moonshot"


@dataclass(frozen=True, slots=True)
class LLMProviderField:
    """Описывает дополнительные поля конфигурации провайдера помимо API-ключа, Base URL и имени модели."""

    config_suffix: str
    label_key: str
    required: bool = False
    secret: bool = False
    default_value: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderEndpoint:
    """Описывает сопутствующие точки входа и адреса API одного провайдера в разных сервисных регионах."""

    endpoint_id: str
    default_label: str
    base_url: str
    api_key_url: str
    model_docs_url: str = ""


@dataclass(frozen=True, slots=True)
class LLMProviderSpec:
    """
    Централизованное объявление LLM-провайдера.

    Здесь собраны устойчивые метаданные, которыми пользуются WebUI, загрузка
    конфигурации и вызовы сервисов, включая отображаемое имя по умолчанию и ключ
    локали; сами переводы тут не хранятся и запросы к API не выполняются. Так
    «что такое провайдер» описывает Registry, а «как его вызывать» по-прежнему
    отвечает адаптер в сервисном слое.
    """

    provider_id: str
    default_label: str
    adapter: str = "openai_compatible"
    api_key_url: str = ""
    default_model: str = ""
    default_base_url: str = ""
    requires_api_key: bool = True
    requires_model_name: bool = True
    requires_base_url: bool = True
    show_api_key: bool = True
    show_base_url: bool = True
    deprecated_models: tuple[str, ...] = ()
    deprecated_base_urls: tuple[str, ...] = ()
    extra_fields: tuple[LLMProviderField, ...] = ()
    service_endpoints: tuple[LLMProviderEndpoint, ...] = ()
    default_service_endpoint_id: str = ""
    international_service_endpoint_id: str = ""

    @property
    def label_key(self) -> str:
        return f"llm_provider_label.{self.provider_id}"

    @property
    def tips_key(self) -> str:
        return f"llm_provider_tips.{self.provider_id}"

    @property
    def endpoint_selector_label_key(self) -> str:
        return f"llm_provider_endpoint_selector.{self.provider_id}"

    @property
    def endpoint_selector_help_key(self) -> str:
        return f"llm_provider_endpoint_selector_help.{self.provider_id}"

    @property
    def authentication_error_key(self) -> str:
        return f"llm_provider_authentication_error.{self.provider_id}"

    def endpoint_label_key(self, endpoint_id: str) -> str:
        return f"llm_provider_endpoint.{self.provider_id}.{endpoint_id}"

    def config_key(self, suffix: str) -> str:
        return f"{self.provider_id}_{suffix}"

    def resolve_model_name(self, configured_model: str | None) -> str:
        """Приводит пустое значение или устаревшее историческое умолчание к текущей модели по умолчанию."""
        model_name = (configured_model or "").strip()
        if not model_name or model_name in self.deprecated_models:
            return self.default_model
        return model_name

    def resolve_base_url(self, configured_base_url: str | None) -> str:
        """Разбирает Base URL и переводит выведенные из эксплуатации исторические адреса на текущее значение по умолчанию."""
        base_url = (configured_base_url or "").strip()
        deprecated_urls = {url.rstrip("/") for url in self.deprecated_base_urls}
        if not base_url or base_url.rstrip("/") in deprecated_urls:
            return self.effective_default_base_url
        return base_url

    def get_service_endpoint(self, endpoint_id: str) -> LLMProviderEndpoint | None:
        """Получает сервисный регион по устойчивому ID, чтобы бизнес-логика не зависела от меняющихся промоссылок."""
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.endpoint_id == endpoint_id
            ),
            None,
        )

    @property
    def default_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Возвращает сервисный регион по умолчанию, объявленный провайдером."""
        return self.get_service_endpoint(self.default_service_endpoint_id)

    @property
    def international_service_endpoint(self) -> LLMProviderEndpoint | None:
        """Возвращает международный сервисный регион, объявленный провайдером."""
        return self.get_service_endpoint(self.international_service_endpoint_id)

    @property
    def effective_default_base_url(self) -> str:
        """Сначала берёт Base URL из региона по умолчанию; у обычных провайдеров используется прежнее поле."""
        endpoint = self.default_service_endpoint
        return endpoint.base_url if endpoint else self.default_base_url

    def preferred_service_endpoint(
        self, *, prefer_international: bool
    ) -> LLMProviderEndpoint | None:
        """Возвращает предпочтительную точку входа по региону интерфейса, безопасно откатываясь к точке по умолчанию, если международной нет."""
        if prefer_international and self.international_service_endpoint:
            return self.international_service_endpoint
        return self.default_service_endpoint

    def effective_api_key_url(self, *, prefer_international: bool = False) -> str:
        """Единообразно определяет страницу получения API-ключа, чтобы endpoint-провайдеры не дублировали ссылки."""
        endpoint = self.preferred_service_endpoint(
            prefer_international=prefer_international
        )
        return endpoint.api_key_url if endpoint else self.api_key_url

    def find_service_endpoint(
        self, configured_base_url: str | None
    ) -> LLMProviderEndpoint | None:
        """Определяет стандартный сервисный регион провайдера по сохранённому Base URL."""
        normalized_url = (configured_base_url or "").strip().rstrip("/")
        if not normalized_url:
            return None
        return next(
            (
                endpoint
                for endpoint in self.service_endpoints
                if endpoint.base_url.rstrip("/") == normalized_url
            ),
            None,
        )

    def select_service_endpoint(
        self,
        configured_base_url: str | None,
        *,
        has_api_key: bool,
        prefer_international: bool,
    ) -> LLMProviderEndpoint | None:
        """
        Выбирает стандартный сервисный регион для показа в WebUI.

        Приоритет у явно сохранённого стандартного адреса; неизвестный адрес
        остаётся пользовательским. В старых конфигурациях мог быть только
        API-ключ без Base URL — такие пользователи продолжают работать с
        регионом по умолчанию из Registry, чтобы обновление не переключило
        сервис из-за другого языка интерфейса. Международная точка входа
        выбирается по языку интерфейса только для полностью новых конфигураций.
        """
        configured_url = (configured_base_url or "").strip()
        if configured_url:
            return self.find_service_endpoint(configured_url)

        default_endpoint = self.default_service_endpoint
        if has_api_key or not prefer_international:
            return default_endpoint

        return self.preferred_service_endpoint(
            prefer_international=prefer_international
        )


# Порядок кортежа задаёт порядок выпадающего списка в WebUI. Чтобы добавить
# обычного OpenAI-совместимого провайдера, обычно достаточно вписать сюда ещё
# один элемент и добавить локаль. Отдельная реализация адаптера в
# app/services/llm.py нужна только провайдерам с другим протоколом.
LLM_PROVIDER_REGISTRY = (
    # Рекомендуемые провайдеры
    LLMProviderSpec(
        "moonshot",
        "Kimi / Moonshot AI",
        default_model="kimi-k3",
        service_endpoints=(
            LLMProviderEndpoint(
                endpoint_id="china",
                default_label="China",
                base_url="https://api.moonshot.cn/v1",
                api_key_url=(
                    "https://platform.kimi.com?"
                    "track_id=track-2f5441d6ffd84c509dd079d78e9db5dc&"
                    "aff=moneyprinterturbo"
                ),
                model_docs_url=(
                    "https://platform.kimi.com/docs/models?"
                    "track_id=track-2f5441d6ffd84c509dd079d78e9db5dc&"
                    "aff=moneyprinterturbo"
                ),
            ),
            LLMProviderEndpoint(
                endpoint_id="global",
                default_label="Global",
                base_url="https://api.moonshot.ai/v1",
                api_key_url=(
                    "https://platform.kimi.ai?"
                    "track_id=track-f6b0a640d35c41deb03b247242a1058c&"
                    "aff=moneyprinterturbo"
                ),
                model_docs_url=(
                    "https://platform.kimi.ai/docs/models?"
                    "track_id=track-f6b0a640d35c41deb03b247242a1058c&"
                    "aff=moneyprinterturbo"
                ),
            ),
        ),
        default_service_endpoint_id="china",
        international_service_endpoint_id="global",
    ),
    # Разработчики популярных моделей и облачные провайдеры
    LLMProviderSpec(
        "openai",
        "OpenAI",
        api_key_url="https://platform.openai.com/api-keys",
        default_model="gpt-5.5",
        default_base_url="https://api.openai.com/v1",
    ),
    LLMProviderSpec(
        "anthropic",
        "Anthropic Claude",
        api_key_url="https://platform.claude.com/settings/keys",
        default_model="claude-sonnet-5",
        default_base_url="https://api.anthropic.com/v1/",
    ),
    LLMProviderSpec(
        "gemini",
        "Google Gemini",
        adapter="gemini",
        api_key_url="https://aistudio.google.com/app/apikey",
        default_model="gemini-3.1-pro-preview",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("gemini-pro", "gemini-1.0-pro"),
    ),
    LLMProviderSpec(
        "deepseek",
        "DeepSeek",
        api_key_url="https://platform.deepseek.com/api_keys",
        default_model="deepseek-v4-pro",
        default_base_url="https://api.deepseek.com",
    ),
    LLMProviderSpec(
        "qwen",
        "Alibaba Cloud Qwen",
        adapter="qwen",
        api_key_url="https://dashscope.console.aliyun.com/apiKey",
        default_model="qwen-max",
        requires_base_url=False,
        show_base_url=False,
    ),
    LLMProviderSpec(
        "azure",
        "Microsoft Azure OpenAI",
        adapter="azure",
        api_key_url=(
            "https://portal.azure.com/#view/"
            "Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI"
        ),
        default_model="gpt-35-turbo",
    ),
    LLMProviderSpec(
        "volcengine",
        "ByteDance VolcEngine Ark",
        api_key_url=(
            "https://www.volcengine.com/activity/ai618?utm_campaign=hw&"
            "utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&"
            "utm_term=MoneyPrinterTurbo"
        ),
        default_model="doubao-seed-2-1-turbo-260628",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    ),
    LLMProviderSpec(
        "grok",
        "xAI Grok",
        api_key_url="https://console.x.ai/",
        default_model="grok-4.3",
        default_base_url="https://api.x.ai/v1",
    ),
    LLMProviderSpec(
        "minimax",
        "MiniMax",
        api_key_url="https://platform.minimax.io/",
        default_model="MiniMax-M3",
        default_base_url="https://api.minimax.io/v1",
    ),
    LLMProviderSpec(
        "mimo",
        "Xiaomi MiMo",
        api_key_url=(
            "https://platform.xiaomimimo.com/docs/zh-CN/quick-start/first-api-call"
        ),
        default_model="mimo-v2.5-pro",
        default_base_url="https://api.xiaomimimo.com/v1",
    ),
    # Агрегаторы и платформы единого доступа
    LLMProviderSpec(
        "shengsuanyun",
        "Shengsuan Cloud",
        api_key_url="https://www.shengsuanyun.com/?from=CH_XUQ4OTSK",
        default_model="deepseek/deepseek-v4-flash",
        default_base_url="https://router.shengsuanyun.com/api/v1",
    ),
    # APIMart предоставляет и бизнес-эндпоинт `/api/v1`, и OpenAI-совместимый `/v1`.
    # Текущий сервисный слой LLM читает choices напрямую через OpenAI SDK, поэтому
    # нужен именно `/v1` без внешней обёртки code/data — адрес асинхронного бизнес-эндпоинта сюда не подходит.
    LLMProviderSpec(
        "apimart",
        "APIMart",
        api_key_url="https://go.apimart.ai/gh-moneyprinterturbo",
        default_model="gpt-5.6-terra",
        default_base_url="https://api.apimart.ai/v1",
    ),
    LLMProviderSpec(
        "cloudflare",
        "Cloudflare AI Gateway",
        adapter="cloudflare_ai_gateway",
        api_key_url="https://dash.cloudflare.com/",
        default_model="openai/gpt-4.1-mini",
        requires_base_url=False,
        show_base_url=False,
        deprecated_models=("@cf/meta/llama-3.1-8b-instruct",),
        extra_fields=(
            LLMProviderField("account_id", "Account ID", required=True),
            LLMProviderField(
                "gateway_id",
                "Gateway ID",
                default_value="default",
            ),
        ),
    ),
    LLMProviderSpec(
        "modelscope",
        "Alibaba ModelScope",
        adapter="modelscope",
        api_key_url=("https://modelscope.cn/docs/model-service/API-Inference/intro"),
        default_model="ZhipuAI/GLM-5.2",
        default_base_url="https://api-inference.modelscope.cn/v1/",
    ),
    LLMProviderSpec(
        "aihubmix",
        "AIHubMix",
        api_key_url="https://aihubmix.com/",
        default_model="gpt-5.4-mini",
        default_base_url="https://aihubmix.com/v1",
    ),
    LLMProviderSpec(
        "aimlapi",
        "AIML API",
        api_key_url="https://aimlapi.com/app/keys",
        default_model="openai/gpt-5-5",
        default_base_url="https://api.aimlapi.com/v1",
    ),
    LLMProviderSpec(
        "evolink",
        "EvoLink",
        api_key_url="https://evolink.ai/dashboard/keys",
        default_model="gpt-5.5",
        default_base_url="https://direct.evolink.ai/v1",
    ),
    LLMProviderSpec(
        "openrouter",
        "OpenRouter",
        api_key_url="https://openrouter.ai/settings/keys",
        default_model="minimax/minimax-m3:free",
        default_base_url="https://openrouter.ai/api/v1",
    ),
    # Локальные развёртывания и универсальные шлюзы
    LLMProviderSpec(
        "ollama",
        "Ollama",
        requires_api_key=False,
        show_api_key=False,
    ),
    LLMProviderSpec(
        "oneapi",
        "OneAPI",
        api_key_url="https://github.com/songquanpeng/one-api",
    ),
    LLMProviderSpec(
        "litellm",
        "LiteLLM",
        adapter="litellm",
        default_model="openai/gpt-4o-mini",
        requires_api_key=False,
        requires_base_url=False,
        show_api_key=False,
        show_base_url=False,
    ),
    # Прочие сервисы инференса и публичные сервисы
    LLMProviderSpec(
        "groq",
        "Groq",
        api_key_url="https://console.groq.com/keys",
        default_model="llama-3.3-70b-versatile",
        default_base_url="https://api.groq.com/openai/v1",
    ),
    LLMProviderSpec(
        "pollinations",
        "Pollinations AI",
        api_key_url="https://enter.pollinations.ai/",
        default_model="openai-fast",
        default_base_url="https://gen.pollinations.ai/v1",
        deprecated_models=("default",),
        deprecated_base_urls=("https://text.pollinations.ai/openai",),
    ),
)

LLM_PROVIDERS = {provider.provider_id: provider for provider in LLM_PROVIDER_REGISTRY}

if len(LLM_PROVIDERS) != len(LLM_PROVIDER_REGISTRY):
    raise RuntimeError("duplicate LLM provider id in registry")


def get_llm_provider(provider_id: str) -> LLMProviderSpec | None:
    return LLM_PROVIDERS.get((provider_id or "").lower())


def normalize_provider_override(value: str | None, default_value: str | None) -> str:
    """
    Оставляет только те пользовательские значения, которые отличаются от
    умолчаний Registry.

    WebUI должен показывать значения по умолчанию в полях ввода, но фиксировать
    их в config.toml из-за этого нельзя: иначе при будущем обновлении модели или
    адреса по умолчанию в Registry старая конфигурация продолжит перекрывать
    новое умолчание.
    """
    normalized_value = (value or "").strip()
    normalized_default = (default_value or "").strip()
    if normalized_value == normalized_default:
        return ""
    return normalized_value
