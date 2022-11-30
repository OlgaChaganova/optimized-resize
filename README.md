# Optimized resize of binary images

Реализация nearest neighbour interpolation и bilinear interpolation для ресайза бинарных изображений на Numpy.


**Подготовка среды (python 3.10.8)**:
```commandline
pip install -r requirements.txt
```

**Генерация бинарных изображений для теста**:

 ```
python src/utils/generate_images.py
 ```

**Запустить программу**:

 ```
python src/main.py <path_to_input_img> <img_width> <img_height> --mode [naive_nearest, vectorized_nearest, naive_bilinear, vectorized_bilinear]
 ```

**Тестирование скорости работы**:

```commandline
python src/test.py
```


---
### Примеры работы

#### Исходное изображение №1 (303x384)

![img](tests/images/1.jpg "Размер: 303x384") 

- [bilinear 124x124](tests/images/resized/1%5B124,%20124%5D-bilinear.jpg)
- [nearest 124x124](tests/images/resized/1%5B124,%20124%5D-nearest.jpg)
- [bilinear 512x512](tests/images/resized/1%5B512,%20512%5D-bilinear.jpg)
- [nearest 512x512](tests/images/resized/1%5B512,%20512%5D-nearest.jpg)

#### Исходное изображение №2 (512x512)

![img](tests/images/2.jpg "Размер: 512x512") 

- [bilinear 256x256](tests/images/resized/2%5B256,%20256%5D-bilinear.jpg)
- [nearest 256x256](tests/images/resized/2%5B256,%20256%5D-nearest.jpg)
- [bilinear 1024x1024](tests/images/resized/2%5B1024,%201024%5D-bilinear.jpg)
- [nearest 1024x1024](tests/images/resized/2%5B1024,%201024%5D-nearest.jpg)


#### Исходное изображение №3 (328x400)

![img](tests/images/3.jpg "Размер: 328x400") 

- [bilinear 124x124](tests/images/resized/3%5B124,%20124%5D-bilinear.jpg)
- [nearest 124x124](tests/images/resized/3%5B124,%20124%5D-nearest.jpg)
- [bilinear 512x512](tests/images/resized/3%5B512,%20512%5D-bilinear.jpg)
- [nearest 512x512](tests/images/resized/3%5B512,%20512%5D-nearest.jpg)