# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ------------Q1---------------
# 1.a
puppy_img = cv2.imread("../given_data/puppy.jpg")
gray_puppy = cv2.cvtColor(puppy_img, cv2.COLOR_BGR2GRAY)
uint8_gray_puppy = np.uint8(gray_puppy)
plt.imshow(uint8_gray_puppy, cmap='gray')
plt.title("Grayscale image of puppy")
plt.show()

# 1.b
def show_histogram(img):
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 6))
    plt.plot(histogram, color='black')
    plt.title('Grayscale Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


show_histogram(uint8_gray_puppy)


# 1.c
def gamma_correction(img, gamma):
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
    gamma_img: An output grayscale image after gamma correction -
    uint8 ndarray of size [H x W x 1].
    """
    # ====== YOUR CODE: ======
    gamma_img = ((img / 255) ** gamma) * 255
    gamma_img = np.uint8(gamma_img)
    # ========================
    return gamma_img


puppy_with_gamma_1 = gamma_correction(uint8_gray_puppy, gamma=0.5)
plt.imshow(puppy_with_gamma_1, cmap='gray')
plt.title("Grayscale image of puppy after gamma correction of 0.5")
plt.show()
show_histogram(puppy_with_gamma_1)

puppy_with_gamma_2 = gamma_correction(uint8_gray_puppy, gamma=1.5)
plt.imshow(puppy_with_gamma_2, cmap='gray')
plt.title("Grayscale image of puppy after gamma correction of 1.5")
plt.show()
show_histogram(puppy_with_gamma_2)






# ---------------------------Q2------------------------------------
# 2.a
def video_to_frames(vid_path: str, start_second, end_second):
    """
    Load a video and return its frames from the wanted time range.
    :param vid_path: video file path.
    :param start_second: time of first frame to be taken from the
    video in seconds.
    :param end_second: time of last frame to be taken from the
    video in seconds.
    :return:
    frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
    containing the wanted video frames.
    """
    # ====== YOUR CODE: ======
    frame_set = []
    video_capture = cv2.VideoCapture(vid_path)
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    if start_second is not None:
        start_frame = int(start_second * frame_rate)
    else:
        start_frame = 0
    if end_second is not None:
        end_frame = int(end_second * frame_rate)
    else:
        end_frame = 0

    frame_count = start_frame
    ret = 1
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while ret and frame_count <= end_frame:
        ret, frame = video_capture.read()
        resized_frame = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
        frame_set.append(resized_frame)
        frame_count += 1

    video_capture.release()
    frame_set = np.array(frame_set)
    # ========================
    return frame_set


# 2.b
def match_corr(corr_obj, img):
    """
    return the center coordinates of the location of 'corr_obj' in 'img'.
    :param corr_obj: 2D numpy array of size [H_obj x W_obj]
    containing an image of a component.
    :param img: 2D numpy array of size [H_img x W_img]
    where H_img >= H_obj and W_img>=W_obj,
    containing an image with the 'corr_obj' component in it.
    :return:
    match_coord: the two center coordinates in 'img'
    of the 'corr_obj' component.
    """
    # ====== YOUR CODE: ======
    max_obj = np.max(cv2.filter2D(corr_obj, 4, corr_obj, borderType=cv2.BORDER_CONSTANT))
    img_corr = cv2.filter2D(img, 4, corr_obj, borderType=cv2.BORDER_CONSTANT)
    dist_matrix = np.abs(max_obj - img_corr)
    min_dist_flat_index = np.argmin(dist_matrix)
    match_coord = np.unravel_index(min_dist_flat_index, dist_matrix.shape)
    # ========================
    return match_coord


# 2.c
vid_to_frames = video_to_frames("../given_data/Corsica.mp4", 250, 260)
gray_corsica = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid_to_frames])
lower_height = (gray_corsica.shape[1]) // 3
corsica_after_cut = gray_corsica[:, lower_height:, 7:627]

# 2.d
fixed_corsica = np.stack([frame for frame in corsica_after_cut])
panorama_array = np.zeros((fixed_corsica.shape[1], int(fixed_corsica.shape[2] * 2.5)))
our_frame = fixed_corsica[125]

start_x = (panorama_array.shape[1] - our_frame.shape[1]) // 2
start_y = (panorama_array.shape[0] - our_frame.shape[0]) // 2
end_x = start_x + our_frame.shape[1]
end_y = start_y + our_frame.shape[0]
panorama_array[start_y:end_y, start_x:end_x] = our_frame

# uint8_gray_frame = np.uint8(our_frame)
# plt.imshow(uint8_gray_frame, cmap='gray')
# plt.title("Original frame we chose")
# plt.show()

# uint8_gray_panorama = np.uint8(panorama_array)
# plt.imshow(uint8_gray_panorama, cmap='gray')
# plt.title("Panorama with our frame in the middle")
# plt.show()

early_frame, later_frame = fixed_corsica[50], fixed_corsica[200]

uint8_gray_early_frame = np.uint8(early_frame)
plt.imshow(uint8_gray_early_frame, cmap='gray')
plt.title("early frame we chose")
plt.show()

uint8_gray_later_frame = np.uint8(later_frame)
plt.imshow(uint8_gray_later_frame, cmap='gray')
plt.title("later frame we chose")
plt.show()

# 2.e
early_rectangle = early_frame[:, 0:100]
later_rectangle = later_frame[:, 520:620]

corr_with_early = match_corr(early_rectangle, panorama_array)
corr_with_later = match_corr(later_rectangle, panorama_array)

uint8_gray_early_rec = np.uint8(early_rectangle)
plt.imshow(uint8_gray_early_rec, cmap='gray')
plt.title(corr_with_early)
plt.show()

uint8_gray_later_rec = np.uint8(later_rectangle)
plt.imshow(uint8_gray_later_rec, cmap='gray')
plt.title(corr_with_later)
plt.show()

# 2.f
# TODO - understand how to calc the avg of the areas where the mid img and the later/early imgs are in the same place
# TODO - in the avg do copy to the original panorama
top_left_early = (corr_with_early[1] - 50, max(0, corr_with_early[0] - 120))
bottom_right_early = (min(1550, corr_with_early[1] + 570), min(240, corr_with_early[0] + 120))

top_left_later = (max(0, corr_with_later[1] - 570), max(0, corr_with_later[0] - 120))
bottom_right_later = (min(1550, corr_with_later[1] + 50), min(240, corr_with_later[0] + 120))

early_size = (bottom_right_early[0] - end_x, 240)
later_size = (start_x - top_left_later[0], 240)

later_mid_area_size = (bottom_right_later[0] - start_x, 240)
early_mid_area_size = (end_x - top_left_early[0], 240)

# panorama_array[start_y:end_y, start_x:end_x] = our_frame
panorama_copy = panorama_array.copy()
#
# panorama_array[top_left_later[1]:240, top_left_later[0]:start_x] = \
#     later_frame[0:later_size[1], 0:later_size[0]]  # only later img
#
# panorama_array[start_y:bottom_right_later[1], start_x:bottom_right_later[0]] = \
#     (panorama_copy[start_y:bottom_right_later[1], start_x:bottom_right_later[0]] +
#      later_frame[later_size[1]:later_size[1]+later_mid_area_size[1],
#      later_size[0]:(later_size[0]+later_mid_area_size[0])]) // 2  # avg between later img and mid img
#
# panorama_array[top_left_early[1]:end_y, top_left_early[0]:end_x] = (panorama_copy[top_left_early[1]:end_y,
#                                                                     top_left_early[0]:end_x] +
#                                                                     early_frame[0:early_mid_area_size[1],
#                                                                     0:early_mid_area_size[0]]) // 2 # avg between early and mid
#
# panorama_array[0:bottom_right_early[1], end_x:bottom_right_early[0]] = \
#     early_frame[0:bottom_right_early[1],
#     early_mid_area_size[0]:bottom_right_early[0]]  # only early img

# panorama_array[top_left_early[1]:bottom_right_early[1], top_left_early[0]:bottom_right_early[0]] = \
#     early_frame[0:early_size[1], 0:early_size[0]]
# panorama_array[top_left_later[1]:bottom_right_later[1], top_left_later[0]:bottom_right_later[0]] = \
#     later_frame[0:later_size[1], 0:later_size[0]]


panorama_array[0:240, top_left_later[0]:start_x] = \
    later_frame[0:240, 0:later_size[0]]  # only later img

panorama_array[0:240, (start_x):bottom_right_later[0]] = \
    (panorama_copy[0:240, (start_x):bottom_right_later[0]] +
     later_frame[0:240, later_size[0]:]) // 2  # avg between later img and mid img

panorama_array[0:240, top_left_early[0]:(end_x)] = \
    (panorama_copy[0:240, top_left_early[0]:end_x] + early_frame[0:240, :(early_mid_area_size[0])]) // 2  # avg between early and mid

panorama_array[0:240, (end_x):bottom_right_early[0]] = \
    early_frame[0:240, early_mid_area_size[0]:]  # only early img

plt.imshow(panorama_array, cmap="gray")
plt.title("Panorama with 3 images combined")
plt.show()


# -----------------Q3-----------------
# 3.a
keyboard_img = cv2.imread("../given_data/keyboard.jpg")
gray_keyboard = cv2.cvtColor(keyboard_img, cv2.COLOR_BGR2GRAY)
uint8_gray_keyboard = np.uint8(gray_keyboard)
plt.imshow(uint8_gray_keyboard, cmap='gray')
plt.title("Grayscale image of keyboard")
plt.show()

vertical_kernel = np.ones((8, 1), dtype=np.uint8)
horizontal_kernel = np.ones((1, 8), dtype=np.uint8)

vertical_erosion = cv2.erode(uint8_gray_keyboard, vertical_kernel)
horizontal_erosion = cv2.erode(uint8_gray_keyboard, horizontal_kernel)

plt.imshow(vertical_erosion, cmap='gray')
plt.title("vertical_erosion")
plt.show()

plt.imshow(horizontal_erosion, cmap='gray')
plt.title("horizontal_erosion")
plt.show()
#
total_img = cv2.add(vertical_erosion, horizontal_erosion)
plt.imshow(total_img, cmap='gray')
plt.title("sum of 2 imgs after erosion")
plt.show()
#
binary_image = np.where(total_img >= int(0.2 * 255), 1, 0)
plt.imshow(binary_image, cmap='gray')
plt.title("binary_image")
plt.show()

# 3.b
not_image = np.uint8(cv2.bitwise_not(binary_image))
median_blur = cv2.medianBlur(not_image, 9)
plt.imshow(median_blur, cmap='gray')
plt.title("median_blur")
plt.show()

# 3.c
square_kernel = np.ones((8, 8), dtype=np.uint8)
square_erosion = cv2.erode(median_blur, square_kernel)
plt.imshow(square_erosion, cmap='gray')
plt.title("square erosion on median img")
plt.show()

# 3.d
mult_two_imgs = (np.uint8(square_erosion)//255) * uint8_gray_keyboard
plt.imshow(mult_two_imgs, cmap='gray')
plt.title("multiply square erosion with original img")
plt.show()

k_matrix = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_img = cv2.filter2D(mult_two_imgs, -1, k_matrix)
plt.imshow(sharpened_img, cmap='gray')
plt.title("sharpened image")
plt.show()

threshold = int(0.75*255)
binary_sharpened_image = np.array(np.where(sharpened_img >= threshold, 1, 0), dtype=np.uint8)
plt.imshow(binary_sharpened_image, cmap='gray')
plt.title("sharpened image with threshold of 75%")
plt.show()


# ---------------Q4-------------
# 4.a
gordon_frames = video_to_frames('../given_data/Flash Gordon Trailer.mp4', 20, 21)
stacked_frames = np.stack([frame for frame in gordon_frames])
chosen_frame = stacked_frames[7]

green_frame = chosen_frame[:, :, 1]
plt.imshow(green_frame, cmap='gray')
plt.title("green chosen frame after gray scale")
plt.show()

resized_frame = cv2.resize(green_frame, (green_frame.shape[1] // 2, green_frame.shape[0] // 2))


def poisson_noisy_image(X, a):
    """
    Creates a Poisson noisy image.
    :param X: The Original image. np array of size [H x W] and of type uint8.
    :param a: number of photons scalar factor
    :return:
    Y: The noisy image. np array of size [H x W] and of type uint8.
    """
    # ====== YOUR CODE: ======
    img_to_float = np.array(X, dtype=float)
    photons_num = img_to_float * a
    noise = np.random.poisson(photons_num)
    noise_frame = noise / a
    noise_fram_after_clip = np.clip(noise_frame, 0, 255)
    Y = np.array(noise_fram_after_clip, dtype=np.uint8)
    # ========================
    return Y


img_with_poisson_noise = poisson_noisy_image(resized_frame, a=3)
plt.imshow(img_with_poisson_noise, cmap='gray')
plt.title("img after adding poisson noise")
plt.show()


# 4.b

def calc_Gk(lambda_reg, kernel, Xk_bar, Y_bar, X_size):
    Xk_to_mtx = np.reshape(Xk_bar, X_size, 'F')
    Gk = cv2.filter2D(Xk_to_mtx, -1, kernel)
    Gk = cv2.filter2D(Gk, -1, kernel)
    final_Gk = lambda_reg * Gk.flatten('F') + Xk_bar - Y_bar
    return final_Gk


def calc_Mu(Gk, kernel, lambda_reg, X_size):
    transpose = np.transpose(Gk)
    Gk_to_mtx = np.reshape(Gk, X_size, 'F')
    Gk_kernel_conv = cv2.filter2D(Gk_to_mtx, -1, kernel)
    Gk_kernel_conv = cv2.filter2D(Gk_kernel_conv, -1, kernel)
    Gk_kernel_conv_final = Gk_kernel_conv.flatten('F')
    mu_to_return = (transpose @ Gk) / (transpose @ Gk + lambda_reg * transpose @ Gk_kernel_conv_final)
    return mu_to_return.flatten('F')


def calc_Err1(Xk_bar, Y_bar, lambda_reg, kernel, X_size):
    Xk = np.reshape(Xk_bar, X_size, 'F')
    Xk_kernel_conv = cv2.filter2D(Xk, -1, kernel)
    Xk_kernel_conv_final = Xk_kernel_conv.flatten('F')
    err1 = (np.transpose(Xk_bar - Y_bar)) @ (Xk_bar - Y_bar) + \
           lambda_reg * (np.transpose(Xk_kernel_conv_final) @ Xk_kernel_conv_final)
    return err1


def calc_Err2(Xk_bar, X_bar):
    err2 = (np.transpose(Xk_bar - X_bar)) @ (Xk_bar - X_bar)
    return err2


def denoise_by_l2(Y, X, num_iter, lambda_reg):
    """
    L2 image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    D_matrix = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    X_bar = X.flatten('F').astype(float)
    Y_bar = Y.flatten('F').astype(float)
    Xk_bar = Y_bar.copy()
    Err1 = np.zeros(num_iter)
    Err2 = np.zeros(num_iter)
    for index in range(num_iter):
        Gk_bar = calc_Gk(lambda_reg, D_matrix, Xk_bar, Y_bar, X.shape)
        Mu_k = calc_Mu(Gk_bar, D_matrix, lambda_reg, X.shape)
        Xk_bar = Xk_bar - Mu_k * Gk_bar
        Err1[index] = calc_Err1(Xk_bar, Y_bar, lambda_reg, D_matrix, X.shape)
        Err2[index] = calc_Err2(Xk_bar, X_bar)
    Xout = np.reshape(Xk_bar, X.shape, 'F')
    # ========================
    return Xout, Err1, Err2


Xout, Err1, Err2 = denoise_by_l2(img_with_poisson_noise, resized_frame, num_iter=50, lambda_reg=0.5)
plt.imshow(Xout, cmap='gray')
plt.title("img after denoising by l2")
plt.show()

figure = plt.figure()
our_plot = figure.add_subplot(1, 1, 1)
our_plot.plot(list(range(1, 51)), np.log(Err1), label='Err1')
our_plot.plot(list(range(1, 51)), np.log(Err2), label='Err2')
our_plot.legend()
plt.title("Errors Plot")
plt.xlabel("iterations")
plt.ylabel("errors values")
plt.show()

# 4.c

gordon_frames_new = video_to_frames('../given_data/Flash Gordon Trailer.mp4', 38, 39)
stacked_frames_new = np.stack([frame for frame in gordon_frames_new])
chosen_frame_new = stacked_frames_new[7]

green_frame_new = chosen_frame_new[:, :, 1]
plt.imshow(green_frame_new, cmap='gray')
plt.title("NEW green chosen frame after gray scale")
plt.show()

resized_frame_new = cv2.resize(green_frame_new, (green_frame_new.shape[1] // 2, green_frame_new.shape[0] // 2))

new_frame_with_noise = poisson_noisy_image(resized_frame_new, a=3)
plt.imshow(new_frame_with_noise, cmap='gray')
plt.title("NEW img after adding poisson noise")
plt.show()

Xout_new, Err1_new, Err2_new = denoise_by_l2(new_frame_with_noise,resized_frame_new,num_iter=50,lambda_reg=0.5)
plt.imshow(Xout_new, cmap='gray')
plt.title("NEW img after denoising by l2")
plt.show()

figure = plt.figure()
our_new_plot = figure.add_subplot(1, 1, 1)
our_new_plot.plot(list(range(1, 51)), np.log(Err1_new), label='Err1_new')
our_new_plot.plot(list(range(1, 51)), np.log(Err2_new), label='Err2_new')
our_new_plot.legend()
plt.title("NEW Errors Plot")
plt.xlabel("iterations")
plt.ylabel("new errors values")
plt.show()


