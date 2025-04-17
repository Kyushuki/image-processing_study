import numpy as np

def get_levels(num_intervals: int) -> np.ndarray:
    """Возвращает массив с границами интервалов яркости.

    Args:
        num_intervals (int): количество интервалов

    Returns:
        np.ndarray: массив со значениями границ интервалов

    """
    res = np.zeros(num_intervals+1,dtype=int)

    for i in range(1,len(res)):
        res[i] = res[i-1]+(255//num_intervals)

    return res


def get_color_map(num_intervals: int) -> np.ndarray:
    """Возвращает color map.

    Args:
        num_intervals (int): количество интервалов

    Returns:
        np.ndarray: color map

    """
    from matplotlib.pyplot import get_cmap

    p = num_intervals
    res = np.ndarray((p,3))
    colors = get_cmap('plasma')
    
    for i in range(p):
        rgb = colors(i/(p-1))
        res[i] = rgb[:3]
    return res


def get_false_color_img(img: np.ndarray, levels: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """Присваивает пикселям grayscale изображения цвет в соответствии с color map.

    Args:
        img (np.ndarray): изображение
        levels (np.ndarray): границы интервалов яркости
        color_map (np.ndarray): color map

    Returns:
        np.ndarray: цветное изображение

    """
    h, w = img.shape
    res = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            pixel = img[i, j]
            for k in range(1, len(levels)):
                if levels[k-1] <= pixel < levels[k]:
                    res[i, j] = color_map[k-1] 
                    break
    return res
