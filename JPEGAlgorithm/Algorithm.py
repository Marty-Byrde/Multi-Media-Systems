import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import os

def read_image(image_path):
    # Read image and convert it to YCbCr (YUV) color space.
    image = Image.open(image_path)
    image_ycbcr = image.convert('YCbCr')
    img_array = np.array(image_ycbcr, dtype=float)
    return img_array

def chroma_subsampling(Cb, Cr):
    # 4:2:0 Chroma Subsampling
    Cb_subsampled = Cb[::2, ::2]
    Cr_subsampled = Cr[::2, ::2]
    return Cb_subsampled, Cr_subsampled

def process_channel_as_block(channel, blocksize=8, blockFunction = None):
    # Dimensions of the image / chanel
    h, w = channel.shape

    # Add Padding so that the image can be divided into blocks of 8 x 8 without any "leftovers"
    h_pad = (blocksize - (h % blocksize) ) % blocksize
    w_pad = (blocksize - (w % blocksize) ) % blocksize
    channel = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0) # set the values of the padding to 0

    transformedChanel = np.zeros_like(channel, dtype=float)
    for i in range(0, channel.shape[0], blocksize):
        for j in range(0, channel.shape[1], blocksize):
            block = channel[i: i + blocksize, j: j + blocksize]

            if blockFunction is None:
                raise RuntimeError("[Error] Missing block-function that is to be applied on each block!")

            transformedBlock = blockFunction(block)
            transformedChanel[i: i+blocksize, j: j+blocksize] = transformedBlock

    # Remove the overflow-padding that was added at the beginning
    return transformedChanel[:h, :w]


def dct_2d_block(block):
    # Apply the two-dimensional dct transformation to a block of an image
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def eliminate_dct_coefficients(dct_block, removal_percentage):
    """
    Eliminates a certain percentage of the DCT coefficients of a block.

    - removal_percentage: The percentage of coefficients that should be removed.
    """
    coefficients = dct_block.flatten()
    amount_of_coefficients = len(coefficients)

    # Determinate the amount of coefficients that remain
    remaining_coefficients = int(amount_of_coefficients * (1 - removal_percentage))

    # Coefficient "Removal", set to-be-removed coefficients to 0
    coefficients[remaining_coefficients: ] = 0

    return coefficients.reshape(dct_block.shape)


def quantization_per_block(block, quantization_factor):
    """
    Quantize colors per block from 8-bit to Q-bit.
    For Q=8, we have full 256 levels (no reduction).
    For Q<8, we reduce the color depth accordingly.
    """
    levels = 2**quantization_factor  # Number of quantization levels -> 2^Q
    print(f'Quantization levels:', levels)

    # Map block values [0..255] to [0..levels-1] -> reduce colors
    q_block = np.round((block / 255.0) * (levels - 1))
    return q_block

def reverse_quantization_and_dct(block, quantization_factor):
    def dequantize_block(q_block ):
        """
        Reconstruct the block values back to the 0..255 range
        """
        levels = 2**quantization_factor
        # Map [0..levels-1] back to [0..255]
        return (q_block / (levels - 1)) * 255.0

    def reverse_dct_2d_block(dct_block):
        """
        Apply the two-dimensional inverse dct transformation to a block of an image
        """
        return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

    block = dequantize_block(block)
    block = process_channel_as_block(block, blockFunction=reverse_dct_2d_block)

    return block

def chroma_reverse_sampling_420(channel, target_shape):
    h, w = target_shape

    # Repeat each pixel twice horizontally and vertically to reverse the 4:2:0 subsampling
    up_h = np.repeat(channel, 2, axis=0)
    up_hw = np.repeat(up_h, 2, axis=1)
    return up_hw[:h, :w]

def main():
    image_path = "OriginalImage.png"
    img_channels = read_image(image_path)

    Y = img_channels[:, :, 0]
    Cb = img_channels[:, :, 1]
    Cr = img_channels[:, :, 2]

    h, w = Y.shape

    # 4:2:0 Chroma Subsampling
    Cb_sampled, Cr_sampled = chroma_subsampling(Cb, Cr)


    # DCT - with blocks with a size of 8 x 8
    Y_dct = process_channel_as_block(Y, blockFunction=dct_2d_block)
    Cb_dct = process_channel_as_block(Cb_sampled, blockFunction=dct_2d_block)
    Cr_dct = process_channel_as_block(Cr_sampled, blockFunction=dct_2d_block)


    # Eliminate X percent of the DCT coefficients by setting them to zero -> reduce information and thereby the quality of the image
    elimination_percentage = int(input("Enter the percentage of DCT coefficients that should be eliminated (0 - 100): ")) / 100

    def elminate_coeffienct_per_block(block):
        return eliminate_dct_coefficients(block, elimination_percentage)

    Y_dct = process_channel_as_block(Y_dct, blockFunction=elminate_coeffienct_per_block)
    Cb_dct = process_channel_as_block(Cb_dct, blockFunction=elminate_coeffienct_per_block)
    Cr_dct = process_channel_as_block(Cr_dct, blockFunction=elminate_coeffienct_per_block)


    quantization_factor = int(input("Enter the amount of colors that you want to have (1 - 8 bit)"))

    # Quantization per block is equal to applying the quantization function to each channel
    def quantize_block(block):
        return quantization_per_block(block, quantization_factor)

    # Y_quantized = process_channel_as_block(Y_dct, blockFunction=quantize_block)
    # Cb_quantized = process_channel_as_block(Cb_dct, blockFunction=quantize_block)
    # Cr_quantized = process_channel_as_block(Cr_dct, blockFunction=quantize_block)

    Y_quantized = quantization_per_block(Y_dct, quantization_factor)
    Cb_quantized = quantization_per_block(Cb_dct, quantization_factor)
    Cr_quantized = quantization_per_block(Cr_dct, quantization_factor)


    # Reverse Quantization and DCT
    Y_inverse = reverse_quantization_and_dct(Y_quantized, quantization_factor)
    Cb_inverse = reverse_quantization_and_dct(Cb_quantized, quantization_factor)
    Cr_inverse = reverse_quantization_and_dct(Cr_quantized, quantization_factor)

    # Reverse Chroma Subsampling
    Cb_upsampled = chroma_reverse_sampling_420(Cb_inverse, (h, w))
    Cr_upsampled = chroma_reverse_sampling_420(Cr_inverse, (h, w))

    # Basically map the values back to the 0..255 range
    Y_final = np.clip(Y_inverse, 0, 255)
    Cb_final = np.clip(Cb_upsampled, 0, 255)
    Cr_final = np.clip(Cr_upsampled, 0, 255)

    # Combine YUV channels and convert back to RGB
    YCbCr_final = np.dstack((Y_final, Cb_final, Cr_final)).astype(np.uint8)
    outputImage = Image.fromarray(YCbCr_final, 'YCbCr').convert('RGB')

    outputImage.save(f"OutputImage_{elimination_percentage}dct_{quantization_factor}bit.jpg", "JPEG")
    outputImage.save("OutputImage.jpg", "JPEG") # So that the compressed image is updated automatically by the viewer to make the evaluation / comparison faster
    print(f"The image has been saved as 'OutputImage.png' and 'OutputImage_{elimination_percentage}dct_{quantization_factor}bit.jpg'")

main()

