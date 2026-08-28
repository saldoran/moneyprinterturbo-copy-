# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /MoneyPrinterTurbo

# Выставляем права 777 на каталог /MoneyPrinterTurbo
RUN chmod 777 /MoneyPrinterTurbo

ENV PYTHONPATH="/MoneyPrinterTurbo"

# Локальные пользователи по умолчанию продолжают отдавать приоритет китайским зеркалам;
# при публикации образа в GHCR из GitHub Actions используется default, чтобы зарубежный
# runner не завис надолго из-за медленного доступа к китайским зеркалам.
ARG DOCKER_BUILD_MIRROR=china
ARG PIP_USE_OFFICIAL=0

# Установка системных зависимостей обязана удовлетворять двум условиям: в китайском
# окружении сохраняется откат по зеркалам, а если отказали все зеркала, сборка Docker
# должна упасть немедленно. В прежнем цикле последним выполнялся sleep, всегда
# возвращавший 0, из-за чего без git и ffmpeg всё равно получался нерабочий образ.
# Здесь «запись источников», «установка» и «три повтора» разнесены в shell-функции с
# чёткими границами, и продолжать ли работу, решает код возврата функции.
# Все источники используют HTTPS: часть сетевых окружений напрямую блокирует открытый HTTP.
RUN set -u; \
    write_debian_sources() { \
        main_url="$1"; \
        security_url="$2"; \
        printf 'deb %s bullseye main\ndeb %s bullseye-updates main\ndeb %s bullseye-security main\n' \
            "$main_url" "$main_url" "$security_url" > /etc/apt/sources.list; \
        rm -rf /var/lib/apt/lists/*; \
    }; \
    install_system_dependencies() { \
        apt-get update && \
        apt-get install -y --no-install-recommends git ffmpeg; \
    }; \
    retry_system_dependencies() { \
        attempt=1; \
        while [ "$attempt" -le 3 ]; do \
            echo "Attempt $attempt: installing system dependencies"; \
            if install_system_dependencies; then \
                return 0; \
            fi; \
            echo "Attempt $attempt failed" >&2; \
            if [ "$attempt" -lt 3 ]; then \
                echo "Retrying in 5 seconds..." >&2; \
                sleep 5; \
            fi; \
            attempt=$((attempt + 1)); \
        done; \
        return 1; \
    }; \
    if [ "$DOCKER_BUILD_MIRROR" = "china" ]; then \
        write_debian_sources \
            "https://mirrors.aliyun.com/debian" \
            "https://mirrors.aliyun.com/debian-security"; \
        if ! retry_system_dependencies; then \
            echo "Aliyun mirror failed, switching to Tsinghua mirror" >&2; \
            write_debian_sources \
                "https://mirrors.tuna.tsinghua.edu.cn/debian" \
                "https://mirrors.tuna.tsinghua.edu.cn/debian-security"; \
            if ! install_system_dependencies; then \
                echo "Tsinghua mirror failed, switching to default Debian mirror" >&2; \
                write_debian_sources \
                    "https://deb.debian.org/debian" \
                    "https://deb.debian.org/debian-security"; \
                if ! install_system_dependencies; then \
                    echo "Failed to install system dependencies from all configured mirrors" >&2; \
                    exit 1; \
                fi; \
            fi; \
        fi; \
    else \
        echo "Using default Debian mirrors"; \
        write_debian_sources \
            "https://deb.debian.org/debian" \
            "https://deb.debian.org/debian-security"; \
        if ! retry_system_dependencies; then \
            echo "Failed to install system dependencies from the default Debian mirror" >&2; \
            exit 1; \
        fi; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt ./

# Локально приоритет у китайских зеркал PyPI; при публикации в GHCR используется официальный PyPI, чтобы зарубежный runner не тормозил на трансграничном доступе к зеркалу.
RUN if [ "$PIP_USE_OFFICIAL" = "1" ]; then \
        pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt; \
    else \
        pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --retries 3 --timeout 60 -r requirements.txt || \
        pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/ --trusted-host mirrors.tuna.tsinghua.edu.cn --retries 3 --timeout 60 -r requirements.txt || \
        pip install --no-cache-dir --retries 3 --timeout 60 -r requirements.txt; \
    fi

# Now copy the rest of the codebase into the image
COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Внутри контейнера обязательно слушаем 0.0.0.0, а на хосте доступ по-прежнему ограничен
# пробросом порта Docker на 127.0.0.1. browser.serverAddress задаёт лишь адрес, который
# показывает браузер, и server.address не заменяет.
CMD ["streamlit", "run", "./webui/Main.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.serverAddress=127.0.0.1", "--server.enableCORS=True", "--browser.gatherUsageStats=False", "--client.toolbarMode=minimal", "--logger.hideWelcomeMessage=True", "--server.showEmailPrompt=False"]

# 1. Build the Docker image using the following command
# docker build -t moneyprinterturbo .

# 2. Run the Docker container using the following command
## For Linux or MacOS:
# docker run -v $(pwd)/config.toml:/MoneyPrinterTurbo/config.toml -v $(pwd)/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
## For Windows:
# docker run -v ${PWD}/config.toml:/MoneyPrinterTurbo/config.toml -v ${PWD}/storage:/MoneyPrinterTurbo/storage -p 127.0.0.1:8501:8501 moneyprinterturbo
