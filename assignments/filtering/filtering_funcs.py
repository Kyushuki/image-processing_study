import numpy as np


def get_expand_img(img: np.ndarray) -> np.ndarray:
    """Дополняет изображение нулями.

    Args:
        img (np.ndarray): изображение

    Returns:
        np.ndarray: изображение, дополненное нулями

    """

    
    # не даёт желаемого результата
    # p = m//2
    # q = n//2
    # rows = np.zeros((p,img.shape[1]))
    # cols = np.zeros((img.shape[0]+2*p,q))
    # img = np.vstack([rows,img,rows])
    # img = np.hstack([cols,img,cols])
    # return img

    m, n = img.shape

    return np.pad(img, ((0,m),(0,n)), mode='constant', constant_values=0)
    

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
    print(f"Сумма ядра {np.sum(kernel)}")
    return kernel 



def get_filtered_image(img: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Осуществляет фильтрацию изображения в частотной области.

    Args:
        img (np.ndarray): изображение
        kern (np.ndarray): ядро фильтра, используемого для фильтрации

    Returns:
        np.ndarray: отфильтрованное изображение

    """
    image_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kern)

    filtered_image = image_fft * kernel_fft

    filtered_image = np.fft.ifft2(filtered_image)

    p, q = kern.shape
    result = np.real(filtered_image)[-p//2:,-q//2:]
    print(f"Размер исходного изображения (размер ядра // 2){(p//2,q//2)}")
    print(f"Размер отфильрованного необрезанного изображения {img.shape}")
    print(f"Размер отфильрованного обрезанного изображения {result.shape}")

    return result


def high_pass_kernel(kern_size: tuple, cutoff: float, order: int) -> np.ndarray:
    """Возвращает ядро фильтра Баттерворта в частотной области.

    Args:
        kern_size (tuple): размер фильтра
        cutoff (float): частота среза
        order (int): порядок фильтра

    Returns:
        np.ndarray: ядро фильтра

    """
    p, q = kern_size
    kern = np.zeros((p,q))
    
    h = lambda d:   1/(1+(cutoff/d)**(2*order))
    for i in range(p):
        for j in range(q):
            d = np.power(((i-p/2)**2+(j-q/2)**2), 1/2 )
            kern[i][j] = h(d) if d!= 0 else 1
    return kern
    # return kern


def get_img_with_impulse_noise(img: np.ndarray, ratio: float) -> np.ndarray:
    """Добавляет импульсный шум на картинку.

    Args:
        img (np.ndarray): изображение
        ratio (float): доля пикселей, подвергающихся изменению

    Returns:
        np.ndarray: изображение с шумом

    """
    
    numofpixels = int(img.size * ratio)
    img_res = img.copy()
    coord = (np.random.randint(0,img.shape[0],numofpixels),
            np.random.randint(0,img.shape[1],numofpixels))
    
    img_res[coord] = 255
    return img_res


def median_filter(img: np.ndarray, kern_size: tuple) -> np.ndarray:
    """Производит фильтрацию с помощью медианного фильтра.

    Args:
        img (np.ndarray): изображение
        kern_size (tuple): размер ядра

    Returns:
        np.ndarray: отфильтрованное изображение

    """
    p, q = img.shape 

    x, y = kern_size[0]//2, kern_size[1]//2

    img_res = img.copy()

    for i in range(x, p - x):
        for j in range(y, q - y):
            kern = img[i - x: i + x, j - y: j + y]
            img_res[i,j] = np.median(kern)

    return img_res
