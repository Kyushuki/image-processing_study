import numpy as np


def downsample_no_filter(img: np.ndarray,f=2) -> np.ndarray:
    """Уменьшает размер изображения без использования сглаживающего фильтра.

    Args:
        img (np.ndarray): изображение

    Returns:
        np.ndarray: уменьшенное изображение

    """
    h = img.shape[0]
    w = img.shape[1]
    res = img[:h - h%2,:w - w%2]
    return res[::f,::f]


def get_pad_img(img: np.ndarray, zeros_num: int) -> np.ndarray:
    """Осуществляет процесс паддинга.

    Args:
        img (np.ndarray): изображение
        zeros_num (int): количество нулей, которое необходимо дополнить на границах изображения

    Returns:
        np.ndarray: изображение c нулями на границах

    """
    z = 2 * zeros_num
    h = img.shape[0] + z
    w = img.shape[1] + z 

    nulls = np.zeros((h,w))

    nulls[zeros_num:zeros_num+(h-z),zeros_num:zeros_num+(w-z)] = img 

    return nulls


def convolve_2d(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Вычисляет свертку изображения и фильтра.

    Args:
        img (np.ndarray): изображение
        kern (np.ndarray): ядро фильтра

    Returns:
        np.ndarray: результат свертки

    """
    img_padded = get_pad_img(img, zeros_num=kern.shape[0]//2)
    h, w = img_padded.shape 
    k_h, k_w = kern.shape 
    result = np.zeros((h - k_h + 1, w - k_w + 1))
    f_kern = np.flip(kern, axis=(0,1))
    # print(result)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            r_img = img_padded[i:i+k_h,j:j+k_w]
            result[i,j] = np.sum(r_img * f_kern)
    # print(result.shape)
    # print(result)
    return result


def gauss_kernel(kern_size: tuple, sigma: float) -> np.ndarray:
    """Возвращает гауссовское ядро.

    Args:
        kern_size (tuple): размер ядра вдоль осей x и y
        sigma (float): среднеквадратическое отклонение

    Returns:
        np.ndarray: гауссовское ядро

    """
    p, q = kern_size
    kernel = np.zeros((p,q))
    
    h = lambda x,y: np.exp(-(x**2+y**2) / (2 * sigma**2)) / (2 * np.pi * sigma)

    pc, pq = p//2, q//2
    for i in range(p):
        x = i - pc 
        for j in range(q):
            y = j - pq
            kernel[i][j] = h(x,y)
    # print(f"Сумма ядра {np.sum(kernel)}")
    return kernel 


def expand_img(img: np.ndarray, row_col_num: int) -> np.ndarray:
    """Добавляет нулевые строки и столбцы между отсчетами.

    Args:
        img (np.ndarray): изображение
        row_col_num (int): количество нулевых строк и столбцов

    Returns:
        np.ndarray: расширенное изображение

    """
    p,q = img.shape
    k = row_col_num
    h = p*(k+1)
    w = q*(k+1)
    nulls = np.zeros((h,w))
    # for i in range(p):
    #     for j in range(q):
    #         nulls[i*(k+1),j*(k+1)] = img[i,j]
    nulls[::k+1,::k+1] = img
    return nulls


def bilin_kernel(l_param: int):
    """Возвращает ядро фильтра билинейной интерполяции.

    Args:
        l_param (int): размер фильтра (одинаков вдоль обоих измерений)

    """
    kern = np.zeros((l_param,l_param))

    beta = 1/(1 + l_param/2)
    h = lambda x: np.max(1-np.abs(beta*x),0)
    h_k = np.ndarray(l_param)
    for i in range(l_param):
        h_k[i]=h(i-l_param//2)
    for i in range(l_param):
        for j in range(l_param):
            kern[i][j] = h_k[i]*h_k[j]
    return kern
