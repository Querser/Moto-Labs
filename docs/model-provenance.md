# Происхождение моделей и внешних данных

Зафиксировано 19 июля 2026 года. Во время гонки все модели исполняются локально.

| Компонент | Назначение | Имя | Версия | Официальный источник | Источник загрузки | Лицензия | Локальный путь | Пользовательские изменения |
|---|---|---|---|---|---|---|---|---|
| Перечисление камер | Связь имени Windows-камеры с индексом и backend OpenCV, включая виртуальную GoPro Webcam | `cv2-enumerate-cameras` | `1.3.3` | [официальный репозиторий](https://github.com/lukehugh/cv2_enumerate_cameras) | [PyPI 1.3.3](https://pypi.org/project/cv2-enumerate-cameras/1.3.3/) | MIT | `.venv/Lib/site-packages/cv2_enumerate_cameras/`, интеграция `app/camera/discovery.py` | Загружен 2026-07-21, интегрирован без изменения; для Windows явно используется DirectShow, профиль GoPro написан в проекте |
| Детектор движения и обработка изображений | MOG2-кандидаты движения, контуры, perspective warp, preprocessing | OpenCV / `opencv-contrib-python` | `4.10.0.84` | [OpenCV](https://github.com/opencv/opencv), [официальная документация MOG2](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) | [PyPI](https://pypi.org/project/opencv-contrib-python/4.10.0.84/) | Apache-2.0 | `.venv/Lib/site-packages/cv2/`, интеграция `app/vision/detector.py` и `app/vision/regions.py` | Интегрирован из библиотеки, не изменён; параметры и обвязка написаны для проекта; MOG2 не классифицирует объект как мотоцикл |
| Детектор мотоцикла | COCO class id 3 `motorcycle` | YOLOX-Tiny, `yolox_tiny.onnx` | `0.1.1rc0`, вход `1×3×416×416` | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [ONNX-документация](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime) | [официальный release asset](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx) | Код: Apache-2.0; отдельная лицензия весов в release не указана | `models/yolox_tiny.onnx` | Скачан 2026-07-17; не изменён и не конвертирован; SHA-256 `427CC366D34E27FF7A03E2899B5E3671425C262EA2291F88BB942BC1CC70B0F7` |
| Multi-object tracker | Связь bbox и траектория между кадрами | Supervision ByteTrack; алгоритм ByteTrack | пакет `supervision==0.29.1` | [Supervision](https://github.com/roboflow/supervision), [оригинальный ByteTrack](https://github.com/FoundationVision/ByteTrack) | [PyPI 0.29.1](https://pypi.org/project/supervision/0.29.1/) | Supervision: MIT; ByteTrack: MIT | `.venv`, адаптер `app/vision/tracker.py` | Интегрирован из библиотеки без изменения; адаптер написан для проекта |
| OCR | Локальное распознавание цифровой строки | RapidOCR: `ch_PP-OCRv4_det_infer.onnx`, `ch_PP-OCRv4_rec_infer.onnx`, `ch_ppocr_mobile_v2.0_cls_infer.onnx` | пакет `rapidocr-onnxruntime==1.4.4` | [RapidOCR](https://github.com/RapidAI/RapidOCR) | [официальный PyPI wheel 1.4.4](https://pypi.org/project/rapidocr-onnxruntime/1.4.4/) | Код: Apache-2.0; repository указывает copyright моделей Baidu | `.venv/Lib/site-packages/rapidocr_onnxruntime/models/` | Интегрирован без конвертации; два CPU-потока на ONNX-сессию, прямой recognition-only путь для локализованной таблички, bounded recovery 256/480 px, межкадровые кропы и временной консенсус написаны для проекта. SHA-256: det `D2A7720D45A54257208B1E13E36A8479894CB74155A5EFE29462512D42F49DA9`, rec `48FC40F24F6D2A207A2B1091D3437EB3CC3EB6B676DC3EF9C37384005483683B`, cls `E47ACEDF663230F8863FF1AB0E64DD2D82B838FCEB5957146DAB185A89D6215C` |
| Резервный OCR | Повторная локализация маленькой цифровой строки в лучших сохранённых кропах | RapidOCR `en_PP-OCRv5_rec_mobile.onnx` с PP-OCRv4 mobile detector | пакет `rapidocr==3.8.1`, PP-OCRv5 English mobile recognition | [RapidOCR](https://github.com/RapidAI/RapidOCR), [официальный список моделей](https://rapidai.github.io/RapidOCRDocs/latest/model_list/) | [официальный ModelScope asset RapidOCR v3.8.0](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx), модели также включены в официальный PyPI wheel 3.8.1 | Код RapidOCR: Apache-2.0; авторские права OCR-моделей принадлежат Baidu/PaddleOCR | `.venv/Lib/site-packages/rapidocr/models/en_PP-OCRv5_rec_mobile.onnx` | Интегрирован без конвертации и изменения весов; используется только для трёх лучших разновременных fallback-кропов, с фильтром ведущей стороны и пунктуации. SHA-256 rec `C3461ADD59BB4323ECBA96A492AB75E06DDA42467C9E3D0C18DB5D1D21924BE8`, det `D2A7720D45A54257208B1E13E36A8479894CB74155A5EFE29462512D42F49DA9` |
| Точный OCR | Распознавание digits-only по лучшим выбранным кропам | `PP-OCRv6_medium_rec_onnx` | PaddleOCR/PaddleX `3.7.0`, ONNX-вес от 2026-06-11 | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [официальная документация PP-OCRv6](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) | [официальный PaddlePaddle Hugging Face repository](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx) через `TextRecognition` | Apache-2.0 | `models/paddlex/official_models/PP-OCRv6_medium_rec_onnx/inference.onnx` | Скачан 2026-07-22, не конвертирован и не обучался в проекте; добавлены увеличение маленького кропа, строгий ASCII digits-only и многокадровый консенсус. SHA-256 `9C09ABF0957F7968C7586464B7397B84AD2387A0497A351AF40E9ACC71B673BA` |
| Детектор текста на щитке | Perspective crop цифровой строки внутри уже найденной передней области | `PP-OCRv5_mobile_det_onnx` | PaddleOCR/PaddleX `3.7.0` | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [официальная документация text detection](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_detection.html) | Официальный PaddleX model hub через `TextDetection(model_name="PP-OCRv5_mobile_det")` | Apache-2.0 | `models/paddlex/official_models/PP-OCRv5_mobile_det_onnx/inference.onnx` | Скачан 2026-07-22, не изменён; perspective rectification написана в проекте. SHA-256 `A431985659DC921974177A95ADCFBB90FD9E51989A5E04D70D0B75F597B6E61D` |
| Экспериментальная проверка OCR (не production) | Оценка спорных многокадровых OCR-кандидатов | `microsoft/Florence-2-base-ft` | snapshot `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e` (0,23B) | [официальный Microsoft repository](https://huggingface.co/microsoft/Florence-2-base-ft) | Тот же repository, закреплённый commit через `huggingface_hub.snapshot_download` | MIT | `models/florence-2-base-ft/model.safetensors`; экспериментальная среда `.venv-florence/` | Скачан 2026-07-22, не конвертирован, не изменён и не обучался в проекте. После контрольного прогона отключён: не загружается лаунчером и не участвует в production-распознавании. SHA-256 `58757D657FF44051314C8030B68E04CB1BB618CA9A4885418F111F6FB708185A` |
| Inference engine | Выполнение YOLOX и PaddleOCR на NVIDIA GPU с CPU fallback; CoreML/CPU на Apple Silicon | ONNX Runtime | `onnxruntime-gpu==1.23.2` Windows; `onnxruntime>=1.23.2` macOS/Linux | [ONNX Runtime](https://github.com/microsoft/onnxruntime), [CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) | [PyPI GPU 1.23.2](https://pypi.org/project/onnxruntime-gpu/1.23.2/) | MIT | `.venv/Lib/site-packages/onnxruntime/` | Интегрирован без изменения; CUDA 12 библиотеки из официальных extras, порядок providers `CUDA, CPU`; на Apple Silicon `CoreML, CPU` |
| Локализатор таблички | Контрастный прямоугольник, светлая малонасыщенная табличка, perspective warp, ведущая часть траектории | FrontNumberBoardRegionExtractor | `0.7.0` | Этот проект | Не загружался | Лицензия проекта | `app/vision/regions.py` | Написан с нуля; адаптирован по предоставленным роликам и кропу таблички; модели и внешние веса не используются |
| Тестовые данные | Smoke-тест реального детектора, не обучение | `Motocross, motorcyclist.jpg` | Снимок со страницы Commons на 2026-07-17 | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File%3AMotocross%2C_motorcyclist.jpg), автор Zalasem1 | Та же страница | CC BY 4.0 | `tests/assets/motorcycle_cc_by_4.jpg` | Скачан 2026-07-17, не изменён; SHA-256 `74635B4C47FAAE593B752606EE77E1524490F010FC2940ED30F40D9914291C5C` |

### Статус экспериментального VLM-verifier на 22 июля 2026 года

`microsoft/Florence-2-base-ft` остаётся документированным локальным артефактом,
но больше не входит в production-пайплайн. Контрольный прогон не увеличил число
верных распознаваний, а сбой отдельного worker-процесса способен был задержать
завершение видео. Аналогично локально оценённый `Qwen/Qwen3-VL-2B-Instruct`
(revision `89644892e4d85e24eaac8bacfd4f463576704203`, Apache-2.0) правильно прочитал
чистый кроп `306`, но на сложных кропах подтвердил ошибочные `909` и `135`; поэтому
он не интегрирован, не загружается лаунчером и не влияет на результаты. Рабочее
решение использует ограниченные RapidOCR/PP-OCRv6 и межкадровый консенсус.

## Обучение

Обучение и дообучение в этой версии не выполнялось. Официальный COCO-вес уже
обнаруживает класс мотоцикла, а тестовый снимок подтверждает работоспособность
локального ONNX-пути. Единственного изображения недостаточно для честного обучения
или оценки качества номерных табличек.

Если будет собран собственный датасет, для каждого видео/кадра необходимо добавить
URL либо описание собственного происхождения, владельца, разрешение на обработку,
лицензию, дату получения, назначение, локальный путь, checksum и разделение
train/validation/test. Особенно нужны реальные ракурсы 45–60°, высота около 2 м,
скорости 20/30/40/60 км/ч, день/сумерки, блики, грязь и частичные перекрытия.

## Проверка на пользовательском материале 19 июля 2026 года

Скрипт `scripts/analyze_reference.py` обработал все 571 кадр предоставленного MOV
размером 1072×1920, 29,9476 FPS, H.264, 19,067 с. На CPU (`CPUExecutionProvider`)
получено 14,15 обработанного кадра/с, средняя задержка общего пайплайна 64,83 мс,
p95 180,07 мс. Это консервативный повторный замер на одновременно работающей пользовательской
Windows-сессии; обработка всех кадров загруженного файла в веб-приложении дала 11,73 кадра/с.
От вручную отмеченного момента достаточной видимости (5,000 с) до
первой детекции прошло 342,67 мс. Первая рамка цифр появилась на 5,476 с, стабильный
номер `306` — на 5,710 с: 367,31 мс после первой детекции и 233,74 мс после первой
рамки цифр. Отдельный кроп 37×48 распознан как `306` с confidence 0,9949 после
увеличения перед OCR. Полный отчёт: `data/reference/benchmark.json`; таблица кадров:
`data/reference/manifest.csv`; аннотации: `data/reference/annotated/`.

Этот один проход использован только для ручной проверки/валидации, не для обучения.
Он не является независимой оценочной выборкой и не подтверждает качество на
фронтально-верхнем производственном ракурсе, других номерах, скоростях или освещении.

### Дополнительный MP4 с несколькими проездами

Файл `video_2026-07-19_13-50-08.mp4` (848×464, 30 FPS, H.264, 137,067 с) использован
только для проверки и настройки пайплайна, не для обучения. Детектор мотоцикла был
запущен на 1371 кадре с шагом три: получено 1106 детекций за 53,25 с, то есть
25,75 обработанного кадра/с для детектора без полного OCR каждого кандидата.

Оценка версии 0.6.0 использовала 19 продиктованных читаемых номеров и
отрицательные интервалы. Полный проход с порогом 72% занял 594,375 с:
приняты `123`, `113`, `15`, `044`, `72`, ложных принятых чтений нет. После
единственного изменения порога согласия с 72% до 70% повторно обработано всё
затронутое окно 47,2–52,0 с; оно дополнительно подтвердило `004` на трёх кадрах,
сохранило `15`, отвергло конфликтный `012` и не добавило ложных чтений. Поэтому
итог версии 0.6.0 на совокупности проверенных окон — 6/19, precision 1,0000,
recall 0,3158. Полный отчёт до изменения и финальный затронутый интервал:
`evaluation_final_step1.json` и `evaluation_004_012_15_final.json`. Значения
разметки никогда не передаются моделям или runtime.

В версии 0.7.0 финальный быстрый проход по всем размеченным окнам с анализом YOLO
через кадр, сохранением межкадровых OCR-кропов и резервным PP-OCRv5 занял 97,442 с
вместо 594,375 с у исходного полного OCR (ускорение 6,10 раза). Приняты шесть
правильных номеров: `123`, `113`, `004`, `15`, `044`, `72`; ложных принятых чтений
нет, задний `313` отклонён. Precision 1,0000, recall 0,3158 (6/19).

Отдельный непрерывный прогон всех 4112 исходных кадров обработал ролик длительностью
137,067 с за 125,290 с: 2056 YOLO-кадров, 16,410 обработанного YOLO-кадра/с,
отношение wall/source 0,914. Это примерно в 8,62 раза быстрее сообщённого прежнего
времени 18 минут. Артефакты: `artifacts/evaluation_ppocrv5_all.json` и
`artifacts/full_video_benchmark_v070_ppocrv5.json`.

Это не означает, что распознаны все проезды: на других участках ролика номер
занимает слишком мало пикселей, закрыт пылью/людьми либо показан сбоку или сзади.
Ошибочный задний `313` и нестабильный `85` больше не принимаются. Артефакты
проверки находятся в `data/reference/video_2026-07-19_13-50-08/`.
Пользовательский ролик не включён в Git и не является лицензированным публичным
датасетом.

### Контроль версии 0.8.0

Среда: Windows 10/11, Intel Core i5-12500H, NVIDIA GeForce RTX 3070 Ti Laptop
8 ГБ, вход 848×464@30 FPS. После устранения конфликта cuDNN отдельный benchmark
YOLOX-Tiny дал 82,46 кадра/с на CUDA. Полный строгий evaluator всех размеченных
окон (каждый кадр, все допустимые recovery-кропы, не быстрый двухпроходный runtime)
обработал 19 читаемых эталонов за 225,537 с и нашёл 7 правильных. Два ошибочных
одиночных чтения этого прогона (`909`, `7`) после него были заблокированы; отдельный
повтор обоих окон подтвердил 0 ложных записей за 28,807 с. Поэтому нельзя честно
заявлять итоговую recall выше 7/19 или реальную точность на гонке: большая часть
номеров в пережатом боковом ролике занимает слишком мало пикселей. Отчёты:
`evaluation_v080_safe_full.json` и `evaluation_v080_false_positive_guard.json`.

Синтетический тест одновременно движущихся независимых треков подтверждает
интерполированные crossing timestamp `60,025 с` и `60,075 с`, то есть разницу
50 мс. Он проверяет математику/состояние, но не заменяет реальную камеру.

Production-прогон через HTTP API после crossing-only deferred OCR обработал весь
файл 137,067 с за 201,521 с (0,68× длительности, 3571 кадр точных окон во втором
проходе). В SQLite записаны семь правильных номеров из 19 читаемых эталонов и ни
одного номера вне разметки: recall 36,84%, precision 100% на этом единственном
ролике. Результат нельзя переносить на другую камеру или считать гарантией.
Артефакты: `artifacts/e2e_v080_optimized_report.json` и
`artifacts/e2e_v080_optimized_race_36.xlsx`.
