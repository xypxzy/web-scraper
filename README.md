# AI Solution for Automatic Website Enhancement

An automated system for analyzing and improving websites using artificial intelligence.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Results](#example-results)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Description

Many companies lose potential customers due to the inefficiency of their websites. This can be caused by ineffective headlines, poorly placed CTA (Call-to-Action) buttons, or insufficiently optimized site structures. Our AI solution automates website analysis and generates improvement recommendations, allowing for increased conversion rates and enhanced user experience (UX).

## Features

- **Web Page Parsing**: Automatically scans the website and extracts key elements such as headlines, buttons, texts, images, meta tags, and more.
- **HTML Structure Analysis**: Checks the correctness of HTML code and identifies potential issues.
- **SEO Analysis**: Evaluates page metadata and content from a search optimization perspective.
- **Content Analysis**: Uses NLP models to analyze the sentiment and quality of textual content.
- **Recommendation Generation**: Provides specific advice on improving the site, including optimized HTML tags and CSS classes.
- **Reports**: Generates detailed reports with analysis results and recommendations.

## Project Structure

```
project_root/
├── main.py
├── scrapers/
│   ├── __init__.py
│   ├── base_scraper.py
│   ├── beautifulsoup_scraper.py
│   └── selenium_scraper.py
├── analyzers/
│   ├── __init__.py
│   ├── base_analyzer.py
│   ├── html_structure_analyzer.py
│   ├── seo_analyzer.py
│   ├── content_analyzer.py
|   └── text_analyzer.py
├── models/
│   └── (machine learning models)
├── reports/
│   └── (report generation)
├── logs/
│   └── (log files)
├── requirements.txt
└── README.md
```

- **scrapers/**: Modules for parsing and retrieving HTML code of pages.
- **analyzers/**: Modules for analyzing HTML code and extracting elements.
- **models/**: Machine learning models for analysis and recommendation generation.
- **reports/**: Modules for generating reports and presenting results.
- **logs/**: Log files for tracking system operations.

## Installation

### Prerequisites

- Python 3.7 or higher
- Installed `pip`

### Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Contents of `requirements.txt`**

   ```
   beautifulsoup4
   requests
   selenium
   webdriver-manager
   transformers
   torch
   ```

4. **Install the driver for Selenium**

   The driver is installed automatically using `webdriver-manager`; no additional configuration is required.

## Usage

### Running the Main Script

```bash
python main.py
```

### Example Usage in Code

```python
from scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from analyzers.html_structure_analyzer import HTMLStructureAnalyzer
from analyzers.seo_analyzer import SEOAnalyzer
from analyzers.content_analyzer import ContentAnalyzer

def main():
    url = "https://example.com/"

    # Initialize scraper
    scraper = BeautifulSoupScraper()
    soup = scraper.scrape(url)

    if soup:
        # Analyze HTML structure
        structure_analyzer = HTMLStructureAnalyzer()
        analysis_result = structure_analyzer.analyze(soup)

        elements = analysis_result['elements']
        recommendations = analysis_result['recommendations']

        # SEO analysis
        seo_analyzer = SEOAnalyzer()
        seo_recommendations = seo_analyzer.analyze(elements)
        recommendations.extend(seo_recommendations)

        # Content analysis
        content_analyzer = ContentAnalyzer()
        content_recommendations = content_analyzer.analyze(elements)
        recommendations.extend(content_recommendations)

        # Output analysis results
        print("Extracted Elements:")
        for key, value in elements.items():
            print(f"\n{key.upper()}:")
            for item in value:
                print(f"- {item}")

        print("\nRecommendations for Improvement:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("Failed to load the page.")

if __name__ == "__main__":
    main()
```

### Logging Configuration

By default, logs are output to the console. You can configure logging to a file by modifying the configuration in `main.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='logs/app.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
```

## Example Results

### Extracted Elements

```
TITLES:
- Example Domain
- Welcome to Example.com

META:
- description: This is an example description.
- keywords: example, domain

LINKS:
- {'text': 'More information...', 'href': 'https://www.iana.org/domains/example'}

IMAGES:
- {'alt': 'Example Image', 'src': '/images/example.png'}

BUTTONS:
- Sign Up
- Learn More
```

### Recommendations for Improvement

```
- The length of the <title> tag should be between 10 and 70 characters.
- The length of the meta description should be between 50 and 160 characters.
- Add more internal links to improve SEO.
- The headline can be improved for a more positive perception: 'Example Domain'
```

## Requirements

- **Python**: 3.7 or higher
- **Python Libraries**:
  - beautifulsoup4
  - requests
  - selenium
  - webdriver-manager
  - transformers
  - torch

## Contributing

We welcome contributions from the community. If you'd like to contribute:

1. **Fork the repository**

2. **Create a branch for new functionality**

   ```bash
   git checkout -b feature/new-feature
   ```

3. **Make changes and commit them**

   ```bash
   git commit -am 'Added new functionality'
   ```

4. **Push changes to the remote repository**

   ```bash
   git push origin feature/new-feature
   ```

5. **Create a Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# AI-решение для автоматического улучшения сайтов

Автоматизированная система анализа и улучшения веб-сайтов с использованием искусственного интеллекта.

## Оглавление

- [Описание проекта](#описание-проекта)
- [Функциональность](#функциональность)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Использование](#использование)
- [Примеры результатов](#примеры-результатов)
- [Требования](#требования)
- [Вклад в проект](#вклад-в-проект)
- [Лицензия](#лицензия)

---

## Описание проекта

Многие компании теряют потенциальных клиентов из-за неэффективности своих веб-сайтов. Это может быть вызвано неудачными заголовками, плохо расположенными CTA-кнопками или недостаточно оптимизированной структурой сайта. Наше AI-решение автоматизирует анализ веб-сайтов и генерирует рекомендации по улучшению, позволяя повысить конверсию и улучшить пользовательский опыт (UX).

## Функциональность

- **Парсинг веб-страниц**: Автоматическое сканирование сайта и извлечение ключевых элементов (заголовки, кнопки, тексты, изображения, метатеги и др.).
- **Анализ структуры HTML**: Проверка корректности HTML-кода и выявление потенциальных проблем.
- **SEO-анализ**: Оценка метаданных и контента страницы с точки зрения поисковой оптимизации.
- **Анализ контента**: Использование моделей NLP для анализа тональности и качества текстового контента.
- **Генерация рекомендаций**: Предоставление конкретных советов по улучшению сайта, включая оптимизированные HTML-теги и CSS-классы.
- **Отчеты**: Генерация подробных отчетов с результатами анализа и рекомендациями.

## Структура проекта

```
project_root/
├── main.py
├── scrapers/
│   ├── __init__.py
│   ├── base_scraper.py
│   ├── beautifulsoup_scraper.py
│   └── selenium_scraper.py
├── analyzers/
│   ├── __init__.py
│   ├── base_analyzer.py
│   ├── html_structure_analyzer.py
│   ├── seo_analyzer.py
│   └── content_analyzer.py
├── models/
│   └── (модели машинного обучения)
├── reports/
│   └── (генерация отчетов)
├── logs/
│   └── (файлы логов)
├── requirements.txt
└── README.md
```

- **scrapers/**: Модули для парсинга и получения HTML-кода страниц.
- **analyzers/**: Модули для анализа HTML-кода и извлечения элементов.
- **models/**: Модели машинного обучения для анализа и генерации рекомендаций.
- **reports/**: Модули для генерации отчетов и представления результатов.
- **logs/**: Файлы логов для отслеживания работы системы.

## Установка

### Предварительные требования

- Python 3.7 или выше
- Установленный `pip`

### Шаги установки

1. **Клонируйте репозиторий**

   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. **Создайте виртуальное окружение**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Для Linux/MacOS
   venv\Scripts\activate  # Для Windows
   ```

3. **Установите зависимости**

   ```bash
   pip install -r requirements.txt
   ```

   **Содержимое `requirements.txt`**

   ```
   beautifulsoup4
   requests
   selenium
   webdriver-manager
   transformers
   torch
   ```

4. **Установите драйвер для Selenium**

   Драйвер устанавливается автоматически с помощью `webdriver-manager`, дополнительная настройка не требуется.

## Использование

### Запуск основного скрипта

```bash
python main.py
```

### Пример использования в коде

```python
from scrapers.beautifulsoup_scraper import BeautifulSoupScraper
from analyzers.html_structure_analyzer import HTMLStructureAnalyzer
from analyzers.seo_analyzer import SEOAnalyzer
from analyzers.content_analyzer import ContentAnalyzer

def main():
    url = "https://example.com/"

    # Инициализация скрапера
    scraper = BeautifulSoupScraper()
    soup = scraper.scrape(url)

    if soup:
        # Анализ структуры HTML
        structure_analyzer = HTMLStructureAnalyzer()
        analysis_result = structure_analyzer.analyze(soup)

        elements = analysis_result['elements']
        recommendations = analysis_result['recommendations']

        # SEO-анализ
        seo_analyzer = SEOAnalyzer()
        seo_recommendations = seo_analyzer.analyze(elements)
        recommendations.extend(seo_recommendations)

        # Анализ контента
        content_analyzer = ContentAnalyzer()
        content_recommendations = content_analyzer.analyze(elements)
        recommendations.extend(content_recommendations)

        # Вывод результатов анализа
        print("Извлеченные элементы:")
        for key, value in elements.items():
            print(f"\n{key.upper()}:")
            for item in value:
                print(f"- {item}")

        print("\nРекомендации по улучшению:")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("Не удалось загрузить страницу.")

if __name__ == "__main__":
    main()
```

### Настройка логирования

Логи по умолчанию выводятся в консоль. Вы можете настроить логирование в файл, изменив конфигурацию в `main.py`:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    filename='logs/app.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)
```

## Примеры результатов

### Извлеченные элементы

```
TITLES:
- Example Domain
- Welcome to Example.com

META:
- description: This is an example description.
- keywords: example, domain

LINKS:
- {'text': 'More information...', 'href': 'https://www.iana.org/domains/example'}

IMAGES:
- {'alt': 'Example Image', 'src': '/images/example.png'}

BUTTONS:
- Sign Up
- Learn More
```

### Рекомендации по улучшению

```
- Длина тега <title> должна быть от 10 до 70 символов.
- Длина мета-описания должна быть от 50 до 160 символов.
- Добавьте больше внутренних ссылок для улучшения SEO.
- Заголовок может быть улучшен для более позитивного восприятия: 'Example Domain'
```

## Требования

- **Python**: 3.7 или выше
- **Библиотеки Python**:
  - beautifulsoup4
  - requests
  - selenium
  - webdriver-manager
  - transformers
  - torch

## Вклад в проект

Мы приветствуем вклад сообщества в развитие проекта. Если вы хотите внести свой вклад:

1. **Форкните репозиторий**
2. **Создайте ветку для новой функциональности**

   ```bash
   git checkout -b feature/new-feature
   ```

3. **Внесите изменения и закоммитьте их**

   ```bash
   git commit -am 'Добавлена новая функциональность'
   ```

4. **Отправьте изменения в удаленный репозиторий**

   ```bash
   git push origin feature/new-feature
   ```

5. **Создайте Pull Request**

## Лицензия

Этот проект лицензирован под лицензией MIT - подробности см. в файле [LICENSE](LICENSE).

---
