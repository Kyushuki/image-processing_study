# Компьютерные методы обработки изображений

# Установка
Для выполнения домашних заданий вам потребуется установить некоторые библиотеки
### pip
Самый простой способ - установить зависимости из `requirements.txt`
```
pip install -r requirements.txt
```
### Poetry
Более правильным и гибким способом является работа с использованием виртуального окружения. Для управления зависимостями здесь используется библиотека **Poetry**.

Чтобы воспользоваться Poetry необходимо:
- Установить Poetry: `pip install poetry`
- Создать виртуальное окружение для проекта (например, в [VS Code](https://code.visualstudio.com/docs/python/environments), в [PyCharm](https://www.jetbrains.com/help/pycharm/poetry.html))
- Выполнить `poetry install` 

При установке Poetry использует файл `pyproject.toml`, который содержит информацию об используемых зависимостях, линтере/форматтере и т.д. 

Если не создавать виртуальное окружение вручную, то Poetry создаст его за вас (см. [docs](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment))

### Jupyter Notebooks
Работать с Jupyter Notebook можно в разных средах и используя разные инструменты, например:
- Работа с `.ipynb` в [VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [Installing Jupyter](https://jupyter.org/install) (JupyterLab или Jupyter Notebook)
- [Google Colaboratory](https://colab.google/)

# Тестирование
Вы можете воспользоваться тестами из папки `tests` для проверки правильности реализации функций из заданий. Для тестирования всех функций из всех заданий необходимо выполнить 
```
pytest tests
```
Для проверки конкретного задания нужно указать соответствующий файл с тестами, например:
```
pytest tests\test_filtering.py
```
**_Обратите внимание_**, что успешно пройденные тесты сами по себе не означают правильно выполненное задание!
