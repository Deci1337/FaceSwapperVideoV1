+# Инструкция по установке и тестированию (Русский)

Пошаговое руководство по установке и запуску проекта Offline Video Face Swap.

---

## Содержание

1. [Системные требования](#1-системные-требования)
2. [Установка FFmpeg](#2-установка-ffmpeg)
3. [Создание виртуального окружения](#3-создание-виртуального-окружения)
4. [Установка PyTorch с CUDA](#4-установка-pytorch-с-cuda)
5. [Установка зависимостей](#5-установка-зависимостей)
6. [Проверка установки](#6-проверка-установки)
7. [Запуск приложения](#7-запуск-приложения)
8. [Тестирование](#8-тестирование)
9. [Решение проблем](#9-решение-проблем)

---

## 1. Системные требования

### Аппаратные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| **GPU** | NVIDIA GTX 1060 6GB | RTX 3060+ 12GB |
| **RAM** | 8 GB | 16 GB |
| **Диск** | 5 GB свободно | 10 GB свободно |
| **CPU** | 4 ядра | 8 ядер |

### Программные требования

- **ОС:** Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **Python:** 3.9, 3.10 или 3.11 (рекомендуется 3.10)
- **CUDA Toolkit:** 11.8 или 12.1 (для GPU)
- **FFmpeg:** Любая актуальная версия

### Проверка версии Python

```powershell
python --version
```

Должно быть: `Python 3.10.x` (или 3.9.x / 3.11.x)

---

## 2. Установка FFmpeg

FFmpeg необходим для работы с аудио (извлечение и объединение).

### Windows (через Chocolatey)

```powershell
# Если Chocolatey установлен:
choco install ffmpeg -y
```

### Windows (вручную)

1. Скачайте FFmpeg: https://github.com/BtbN/FFmpeg-Builds/releases
   - Выберите файл: `ffmpeg-master-latest-win64-gpl.zip`

2. Распакуйте в `C:\ffmpeg`

3. Добавьте в PATH:
   ```powershell
   # Откройте PowerShell от Администратора
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", "Machine")
   ```

4. Перезапустите терминал

### Проверка установки FFmpeg

```powershell
ffmpeg -version
```

Должно вывести версию FFmpeg (например, `ffmpeg version 6.0`).

---

## 3. Создание виртуального окружения

### Шаг 1: Перейдите в папку проекта

```powershell
cd C:\Users\HONOR\Desktop\FaceDetectorPractice\offline-faceswap
```

### Шаг 2: Создайте виртуальное окружение

```powershell
python -m venv venv
```

### Шаг 3: Активируйте окружение

**Windows PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**Если ошибка "execution policy":**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

После активации в начале строки появится `(venv)`:
```
(venv) PS C:\Users\HONOR\Desktop\FaceDetectorPractice\offline-faceswap>
```

---

## 4. Установка PyTorch с CUDA

PyTorch нужен для работы моделей машинного обучения.

### Определите версию CUDA

```powershell
nvidia-smi
```

В выводе найдите `CUDA Version: XX.X` (например, 12.2).

### Установка PyTorch

**Для CUDA 11.8:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Для CUDA 12.1:**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Без GPU (только CPU, очень медленно):**
```powershell
pip install torch torchvision
```

### Проверка установки PyTorch

```powershell
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Ожидаемый вывод:**
```
PyTorch version: 2.1.0+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
```

---

## 5. Установка зависимостей

```powershell
pip install -r requirements.txt
```

Это установит:
- `insightface` - детекция и распознавание лиц
- `onnxruntime-gpu` - инференс на GPU
- `mediapipe` - отслеживание рук и лица
- `opencv-python` - обработка видео
- `gfpgan` - улучшение качества лица
- `ffmpeg-python` - работа с аудио
- `typer` - CLI интерфейс

**Время установки:** ~5-10 минут

---

## 6. Проверка установки

Выполните все команды по порядку:

### 6.1. Проверка PyTorch и CUDA

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```
Ожидается: `CUDA: True`

### 6.2. Проверка ONNX Runtime

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
Ожидается: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

### 6.3. Проверка MediaPipe

```powershell
python -c "import mediapipe; print('MediaPipe: OK')"
```
Ожидается: `MediaPipe: OK`

### 6.4. Проверка InsightFace

```powershell
python -c "import insightface; print('InsightFace: OK')"
```
Ожидается: `InsightFace: OK`

### 6.5. Проверка CLI

```powershell
python cli.py --help
```

**Ожидаемый вывод:**
```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  Offline Video Face Swap + Landmarks Debug Tool

Options:
  --help  Show this message and exit.

Commands:
  all        Run both landmarks and face-swap pipelines.
  landmarks  Generate debug video with hand/face skeleton overlay.
  swap       Replace faces in video with source face image.
```

---

## 7. Запуск приложения

### 7.1. Режим Landmarks (визуализация скелета)

Создает видео с наложением скелета рук и сетки лица.

```powershell
python cli.py landmarks --input <видео.mp4> --output <результат.mp4>
```

**Параметры:**
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--input` | Входное видео | Обязательный |
| `--output` | Выходное видео | Обязательный |
| `--face-mode` | `mesh` (сетка) или `bbox` (рамка) | `mesh` |
| `--fps` | FPS выходного видео | Как у входного |

**Пример:**
```powershell
python cli.py landmarks --input video.mp4 --output debug_landmarks.mp4 --face-mode mesh
```

---

### 7.2. Режим Swap (замена лица)

Заменяет лицо на видео на лицо с фотографии.

```powershell
python cli.py swap --input <видео.mp4> --source-face <лицо.jpg> --output <результат.mp4>
```

**Параметры:**
| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--input` | Входное видео | Обязательный |
| `--source-face` | Фото лица-донора | Обязательный |
| `--output` | Выходное видео | Обязательный |
| `--quality` | `low`, `medium`, `high` | `high` |
| `--enable-enhancer` | Включить GFPGAN | `False` |
| `--keep-audio` | Сохранить аудио | `True` |
| `--provider` | `cuda` или `cpu` | `cuda` |

**Пример (базовый):**
```powershell
python cli.py swap --input video.mp4 --source-face myface.jpg --output result.mp4
```

**Пример (с улучшением качества):**
```powershell
python cli.py swap --input video.mp4 --source-face myface.jpg --output result_hq.mp4 --enable-enhancer --quality high
```

**Пример (тонкая настройка):**
```powershell
python cli.py swap --input video.mp4 --source-face myface.jpg --output result_tuned.mp4 --enable-enhancer --enhancer-weight 0.5 --color-correction 0.6
```

**Новые параметры для качества:**
- `--enhancer-weight 0.0-1.0` - сила GFPGAN (0.5-0.7 рекомендуется, выше = "пластиковое" лицо)
- `--color-correction 0.0-1.0` - коррекция цвета под освещение видео (0.3-0.6 рекомендуется)

---

### 7.3. Режим All (оба пайплайна)

Запускает landmarks и swap последовательно.

```powershell
python cli.py all --input <видео.mp4> --source-face <лицо.jpg> --output-dir <папка>
```

**Пример:**
```powershell
python cli.py all --input video.mp4 --source-face myface.jpg --output-dir ./output
```

**Результат:**
- `output/debug_landmarks.mp4` - визуализация скелета
- `output/result_faceswap.mp4` - видео с замененным лицом

---

## 8. Тестирование

### Подготовка тестовых файлов

1. **Тестовое видео** (`test.mp4`):
   - Длительность: 5-10 секунд
   - Разрешение: 720p или 1080p
   - Должно быть видно лицо и руки

2. **Фото лица** (`face.jpg`):
   - Четкое фронтальное фото лица
   - Хорошее освещение
   - Размер: минимум 256x256 пикселей

---

### Тест 1: Landmarks (быстрый тест)

```powershell
python cli.py landmarks --input test.mp4 --output test_debug.mp4
```

**Что проверить:**
- [ ] Команда запустилась без ошибок
- [ ] Появился progress bar
- [ ] Создался файл `test_debug.mp4`
- [ ] В видео видны зеленые точки рук и сетка лица

**Ожидаемое время:** ~10-30 секунд для 5-секундного видео

---

### Тест 2: Face Swap (основной тест)

```powershell
python cli.py swap --input test.mp4 --source-face face.jpg --output test_swap.mp4
```

**Что проверить:**
- [ ] Модели загрузились (первый запуск ~500MB скачивания)
- [ ] Появился progress bar
- [ ] Создался файл `test_swap.mp4`
- [ ] Лицо в видео заменено на лицо с фото
- [ ] Аудио сохранено (если было)

**Ожидаемое время:** ~30-60 секунд для 5-секундного видео на RTX 3060

---

### Тест 3: С улучшением GFPGAN

```powershell
python cli.py swap --input test.mp4 --source-face face.jpg --output test_enhanced.mp4 --enable-enhancer
```

**Что проверить:**
- [ ] Скачалась модель GFPGAN (~300MB, первый раз)
- [ ] Качество лица выше, чем без `--enable-enhancer`
- [ ] Нет сильных артефактов ("пластиковое" лицо)

**Ожидаемое время:** ~60-120 секунд (GFPGAN замедляет)

---

### Тест 4: Режим CPU (проверка fallback)

```powershell
python cli.py swap --input test.mp4 --source-face face.jpg --output test_cpu.mp4 --provider cpu
```

**Что проверить:**
- [ ] Работает без GPU
- [ ] Предупреждение о медленной работе

**Ожидаемое время:** ~5-10 минут для 5-секундного видео

---

### Тест 5: Полный пайплайн

```powershell
python cli.py all --input test.mp4 --source-face face.jpg --output-dir ./test_output
```

**Что проверить:**
- [ ] Создалась папка `test_output`
- [ ] Создались оба файла: `debug_landmarks.mp4` и `result_faceswap.mp4`

---

## 9. Решение проблем

### Ошибка: "CUDA not available"

**Причина:** PyTorch установлен без поддержки CUDA или драйверы NVIDIA устарели.

**Решение:**
```powershell
# 1. Проверьте драйвер
nvidia-smi

# 2. Переустановите PyTorch
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Ошибка: "No face detected in source image"

**Причина:** Модель не смогла найти лицо на фото.

**Решение:**
- Используйте четкое фронтальное фото лица
- Убедитесь, что лицо хорошо освещено
- Попробуйте фото большего разрешения (512x512+)

---

### Ошибка: "Could not open video file"

**Причина:** Файл не найден или поврежден.

**Решение:**
```powershell
# Проверьте файл
ffprobe video.mp4

# Убедитесь в правильности пути
Test-Path "video.mp4"
```

---

### Ошибка: "FFmpeg not found"

**Причина:** FFmpeg не установлен или не в PATH.

**Решение:**
```powershell
# Проверьте
ffmpeg -version

# Если не найден - установите (см. раздел 2)
```

---

### Ошибка: "Out of Memory" (OOM)

**Причина:** Недостаточно видеопамяти GPU.

**Решение:**
1. Используйте видео меньшего разрешения (720p вместо 1080p)
2. Закройте другие приложения, использующие GPU
3. Используйте `--provider cpu` (медленно, но работает)

---

### Предупреждение: "CUDA not available, using CPU"

**Причина:** ONNX Runtime не нашел CUDA.

**Решение:**
```powershell
# Проверьте доступные провайдеры
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Если только CPUExecutionProvider:
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.16.0
```

---

## Ожидаемая производительность

| Видео | Разрешение | GPU | Landmarks | Swap | Swap + GFPGAN |
|-------|------------|-----|-----------|------|---------------|
| 10 сек | 720p | RTX 3060 | ~5 сек | ~15 сек | ~30 сек |
| 30 сек | 720p | RTX 3060 | ~15 сек | ~45 сек | ~90 сек |
| 60 сек | 1080p | RTX 3060 | ~45 сек | ~3 мин | ~5 мин |
| 10 сек | 720p | CPU | ~30 сек | ~3 мин | ~6 мин |

---

## Быстрый старт (все команды)

```powershell
# 1. Перейти в папку проекта
cd C:\Users\HONOR\Desktop\FaceDetectorPractice\offline-faceswap

# 2. Создать и активировать venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Установить PyTorch с CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Установить зависимости
pip install -r requirements.txt

# 5. Проверить установку
python cli.py --help

# 6. Запустить тест landmarks
python cli.py landmarks --input test.mp4 --output debug.mp4

# 7. Запустить тест swap
python cli.py swap --input test.mp4 --source-face face.jpg --output result.mp4
```

---

**Готово!** Если возникли вопросы - проверьте раздел "Решение проблем" выше.

