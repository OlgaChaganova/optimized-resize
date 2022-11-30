# Optimized resize

Основано на [статье](https://habr.com/ru/post/340966/).

**Генерация бинарных изображений для теста**:

 ```
python src/utils/generate_images.py
 ```

**Запустить программу**:

 ```
python src/main.py <path_to_input_img> <img_width> <img_height> --mode [naive, vectorized]
 ```

**Тестирование скорости работы**:

```commandline
python test.py
```