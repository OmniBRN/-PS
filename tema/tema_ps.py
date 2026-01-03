import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, datasets
from scipy.fft import dctn, idctn
import sys
import os
import glob
import pickle
import heapq
from collections import Counter

# Std Quantization table (Luminance)
Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]])

# Flattening order (low to high freq)
ZIGZAG_ORDER = [
    0, 1, 5, 6,14,15,27,28,
    2, 4, 7,13,16,26,29,42,
    3, 8,12,17,25,30,41,43,
    9,11,18,24,31,40,44,53,
   10,19,23,32,39,45,52,54,
   20,22,33,38,46,51,55,60,
   21,34,37,47,50,56,59,61,
   35,36,48,49,57,58,62,63
]

def break_image_YCbCr(image):
    """RGB -> YCbCr conversion."""
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    Y = 0 + (0.299 * R) + (0.587 * G) + (0.114 * B)
    Cb = 128 - (0.168736 * R) - (0.331264 * G) + (0.5 * B)
    Cr = 128 + (0.5 * R) - (0.418688 * G) - (0.081312 * B)
    return Y, Cb, Cr

def reassamble_image(Y, Cb, Cr):
    """YCbCr -> RGB conversion."""
    R = Y + 1.402 * (Cr-128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    reassambled_image = np.dstack((R, G, B))
    reassambled_image = np.clip(reassambled_image, 0, 255).astype(np.uint8)
    return reassambled_image

def quantization(X_8x8_segment, quality):
    """Apply DCT and divide by Q matrix."""
    X_8x8_segment_dctn = dctn(X_8x8_segment)
    Q_jpeg_temp = Q_jpeg * (1/quality)
    return np.round(X_8x8_segment_dctn/Q_jpeg_temp)

def dequantization(X_8x8_segment_dctn_q, quality):
    """Mult by Q matrix to reverse."""
    Q_jpeg_temp = Q_jpeg * (1/quality)
    X_8x8_segment_dtcn_jpeg = Q_jpeg_temp * X_8x8_segment_dctn_q
    return X_8x8_segment_dtcn_jpeg

def jpeg_segment(X_8x8_segment, quality):
    """Process single block."""
    X_8x8_segment_dctn_q = quantization(X_8x8_segment, quality)
    X_8x8_segment_dctn_jpeg = dequantization(X_8x8_segment_dctn_q, quality)
    X_8x8_segment_jpeg = idctn(X_8x8_segment_dctn_jpeg)
    return X_8x8_segment_jpeg

def jpeg_image(X, quality):
    """Full image pipeline: Split -> Process -> Merge."""
    segments = break_image_into_segments(X)
    H, W = X.shape
    X_jpeg = np.zeros(shape=X.shape)

    H_segments, H_segments_r = H // 8, H % 8
    W_segments, W_segments_r = W // 8, W % 8

    if H_segments_r != 0: H_segments += 1
    if W_segments_r != 0: W_segments += 1
    
    index = 0
    for i in range(H_segments):
        for j in range(W_segments):
            X_segment = segments[index][0]
            orig_h = segments[index][1][0]
            orig_w = segments[index][1][1]
            processed = jpeg_segment(X_segment, quality)
            
            X_jpeg[8*i : 8*i+orig_h, 8*j : 8*j+orig_w] = processed[:orig_h, :orig_w]
            
            index += 1
    return X_jpeg

def break_image_into_segments(X):
    """
    Slices image into 8x8 blocks.
    Pads edges with reflection or constants based on remainder size.
    """
    H, W = X.shape
    W_segments, W_segments_r = W//8, W%8
    H_segments, H_segments_r = H//8, H%8
    has_W_r = W_segments_r != 0
    has_H_r = H_segments_r != 0

    if W_segments_r != 0:
        W_segments+=1
    if H_segments_r != 0:
        H_segments+=1
    
    segments = []
    for i in range(H_segments):
        for j in range(W_segments):
            X_segment = X[8*i:8*(i+1), 8*j:8*(j+1)]
            segment_h, segment_w = X_segment.shape
            
            # Padding edge cases
            if has_W_r and has_H_r and i == H_segments-1 and j == W_segments-1:
                # Bottom-Right
                mode_w = 'constant' if W_segments_r <=3 else 'reflect'
                mode_h = 'constant' if H_segments_r <=3 else 'reflect'
                if mode_w == mode_h == 'constant':
                    X_segment = np.pad(X_segment, ((0, 8-H_segments_r), (0, 8-W_segments_r)), mode='constant')
                elif mode_w == mode_h == 'reflect':
                    X_segment = np.pad(X_segment, ((0, 8-H_segments_r), (0, 8-W_segments_r)), mode='reflect')
                else:
                    X_segment = np.pad(X_segment, ((0, 8-H_segments_r), (0, 0)), mode=mode_w)
                    X_segment = np.pad(X_segment, ((0, 0), (0, 8-H_segments_r)), mode=mode_h)

            elif has_H_r and i == H_segments-1:
                # Bottom
                mode = 'constant' if W_segments_r <=3 else 'reflect'
                X_segment = np.pad(X_segment, ((0, 8-H_segments_r), (0, 0)), mode=mode)

            elif has_W_r and j == W_segments-1:
                # Right
                mode = 'constant' if H_segments_r <=3 else 'reflect'
                X_segment = np.pad(X_segment, ((0, 0), (0,8-W_segments_r)), mode=mode)
                
            segments.append((X_segment, (segment_h, segment_w)))
    return segments
    
def jpeg_compress_image(image, quality):
    """Compress all channels and rebuild."""
    Y, Cb, Cr = break_image_YCbCr(image)
    Y_c = jpeg_image(Y, quality)
    Cb_c = jpeg_image(Cb, quality)
    Cr_c = jpeg_image(Cr, quality)
    reassambled_image = reassamble_image(Y_c,Cb_c,Cr_c)
    return reassambled_image

def MSE(original, compressed):
    return np.mean((original.astype(float)-compressed.astype(float))**2)

def build_huffman_dict(data):
    """Build Huffman tree using heap."""
    if not data: return {}
    counts = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in counts.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        small = heapq.heappop(heap)
        second_small = heapq.heappop(heap)
        for pair in small[1:]:
            pair[1] = '0' + pair[1]
        for pair in second_small[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [small[0] + second_small[0]] + small[1:] + second_small[1:])
    return {symbol: bitstr for symbol, bitstr in heapq.heappop(heap)[1:]}

def get_quantinized_segments(image, quality):
    """Get raw coeffs for serialization."""
    segments = break_image_into_segments(image)
    all_information = []
    for segment, _ in segments:
        segment_q = quantization(segment, quality)
        zigzaged_segment = segment_q.flatten()[ZIGZAG_ORDER]
        all_information.extend(zigzaged_segment.astype(int).tolist())
    return all_information

def serialize_file(image, quality, filename):
    """Save compressed image to bin."""
    if filename == "output.png": filename = "output.bin"
    Y, Cb, Cr = break_image_YCbCr(image)

    all_data = get_quantinized_segments(Y, quality) + get_quantinized_segments(Cb, quality) + get_quantinized_segments(Cr, quality)
    
    huffman_dictionary = build_huffman_dict(all_data)
    bitstring = "".join(huffman_dictionary[val] for val in all_data)
    
    # Byte alignment
    padding = 8 - (len(bitstring) % 8)
    bitstring += "0" * padding
    
    # Bit packing
    byte_array = bytearray()
    for i in range(0, len(bitstring), 8):
        byte_array.append(int(bitstring[i:i+8], 2))
        
    data = {
        'dims' : image.shape,
        'quality' : quality,
        'huffman_dict' : huffman_dictionary,
        'padding': padding,
        'data': byte_array
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Serialized image to {filename}")

def deserialize_file(filename, output_filename):
    """Load binary and reconstruct."""
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    
    huff_dict = obj['huffman_dict']
    rev_dict = {bitstr: val for val, bitstr in huff_dict.items()}
    
    # Unpack
    bitstring = "".join(bin(byte)[2:].zfill(8) for byte in obj['data'])
    if obj['padding'] > 0:
        bitstring = bitstring[:-obj['padding']]
    
    decoded_values = []
    buffer = ""
    for bit in bitstring:
        buffer += bit
        if buffer in rev_dict:
            decoded_values.append(rev_dict[buffer])
            buffer = ""
            
    total_pixels = len(decoded_values) // 3
    Y_vals = decoded_values[:total_pixels]
    Cb_vals = decoded_values[total_pixels:2*total_pixels]
    Cr_vals = decoded_values[2*total_pixels:]

    def rebuild_channel(vals, dims):
        """Inverse ZigZag + IDCT."""
        H, W = dims[0], dims[1]
        H_segs = (H // 8) + (1 if H % 8 != 0 else 0)
        W_segs = (W // 8) + (1 if W % 8 != 0 else 0)
        out = np.zeros((H_segs*8, W_segs*8))
        
        for idx in range(len(vals)//64):
            block_64 = np.array(vals[idx*64:(idx+1)*64])
            block_8x8 = np.zeros(64)
            for i, zig_idx in enumerate(ZIGZAG_ORDER):
                block_8x8[zig_idx] = block_64[i]
            
            i, j = idx // W_segs, idx % W_segs
            dq = dequantization(block_8x8.reshape(8,8), obj['quality'])
            out[8*i:8*(i+1), 8*j:8*(j+1)] = idctn(dq)
        return out[:H, :W]

    Y_rec = rebuild_channel(Y_vals, obj['dims'])
    Cb_rec = rebuild_channel(Cb_vals, obj['dims'])
    Cr_rec = rebuild_channel(Cr_vals, obj['dims'])
    
    final_img = reassamble_image(Y_rec, Cb_rec, Cr_rec)
    plt.imsave(output_filename, final_img.astype(np.uint8))

def serialize_video(input_path, quality, output_bin):
    """
    Two-pass serialization: 
    1. Count freqs for Huffman
    2. Encode and stream bytes to disk
    Uses standard quantization/segmentation functions.
    """
    os.system(f"mkdir -p temp_frames")
    os.system(f"ffmpeg -loglevel error -i {input_path} temp_frames/f_%04d.png")
    frame_files = sorted(glob.glob("temp_frames/f_*.png"))
    
    if not frame_files:
        print("Error: No frames extracted.")
        return

    first_frame = plt.imread(frame_files[0])
    h, w = first_frame.shape[:2]
    
    def get_frame_coeffs_standard(filepath):
        img = plt.imread(filepath)[:,:,:3]
        if img.max() <= 1.0: img *= 255

        def process_chan(channel):
            segments = break_image_into_segments(channel)
            channel_coeffs = []
            
            for segment, _ in segments:
                q_block = quantization(segment, quality)
                zigzagged = q_block.flatten()[ZIGZAG_ORDER]
                channel_coeffs.extend(zigzagged.astype(int).tolist())
            return channel_coeffs

        Y, Cb, Cr = break_image_YCbCr(img)
        return process_chan(Y) + process_chan(Cb) + process_chan(Cr)

    sample_frames = frame_files[::5] 
    coeffs = []
    for i, filepath in enumerate(sample_frames):
        coeffs.extend(get_frame_coeffs_standard(filepath))
        print(f"\rScanning sample {i+1}/{len(sample_frames)}", end="", flush=True)
    print("\r\033[K", end="", flush=True)

    huff_dict = build_huffman_dict(coeffs)
    header = {
        'dims': (h, w),
        'num_frames': len(frame_files),
        'quality': quality,
        'huffman_dict': huff_dict,
    }
    
    with open(output_bin, 'wb') as f:
        pickle.dump(header, f)

    bit_buffer_str = ""
    with open(output_bin, 'ab') as f:
        for i, filepath in enumerate(frame_files):
            print(f"\rAdding Frame #{i+1}", end="", flush=True)
            coeffs = get_frame_coeffs_standard(filepath)
            frame_bits = "".join(huff_dict.get(val, huff_dict.get(0, "0")) for val in coeffs)
            bit_buffer_str += frame_bits
            if len(bit_buffer_str) >= 8:
                num_bytes = len(bit_buffer_str) // 8
                bits_to_write = bit_buffer_str[:num_bytes*8]
                byte_chunk = int(bits_to_write, 2).to_bytes(num_bytes, byteorder='big')
                f.write(byte_chunk)
                bit_buffer_str = bit_buffer_str[num_bytes*8:]
        if len(bit_buffer_str) > 0:
            padding = 8 - len(bit_buffer_str)
            bit_buffer_str += "0" * padding
            last_byte = int(bit_buffer_str, 2).to_bytes(1, byteorder='big')
            f.write(last_byte)
    os.system("rm -r temp_frames/")
    print(f"\nSaved video: {output_bin}")

def deserialize_video(input_bin, output_filename):
    """Stream decode .bin -> frames -> ffmpeg."""
    print(f"Loading {input_bin}...")
    
    with open(input_bin, 'rb') as f:
        header = pickle.load(f)
        raw_data = f.read()

    h, w = header['dims']
    num_frames = header['num_frames']
    quality = header['quality']
    huff_dict = header['huffman_dict']

    print(f"Decoding {len(raw_data)} bytes...")
    rev_dict = {v: k for k, v in huff_dict.items()}
    bitstring = "".join(f"{byte:08b}" for byte in raw_data)
    decoded_values = []
    buffer = ""
    H_segs = (h // 8) + (1 if h % 8 != 0 else 0)
    W_segs = (w // 8) + (1 if w % 8 != 0 else 0)
    coeffs_per_frame = (H_segs * W_segs * 64) * 3
    total_expected = coeffs_per_frame * num_frames
    
    print("Reconstructing frames...")
    current_coeff_count = 0
    
    # Decode Bitstream
    for bit in bitstring:
        buffer += bit
        if buffer in rev_dict:
            val = rev_dict[buffer]
            decoded_values.append(val)
            buffer = ""
            current_coeff_count += 1
            if current_coeff_count >= total_expected:
                break

    def rebuild_channel(vals, dims):
        H, W = dims
        H_segs = (H // 8) + (1 if H % 8 != 0 else 0)
        W_segs = (W // 8) + (1 if W % 8 != 0 else 0)
        
        out = np.zeros((H_segs*8, W_segs*8))
        for idx in range(len(vals)//64):
            block_64 = np.array(vals[idx*64:(idx+1)*64])
            block_8x8 = np.zeros(64)
            for i, zig_idx in enumerate(ZIGZAG_ORDER):
                block_8x8[zig_idx] = block_64[i]
            block_8x8 = block_8x8.reshape(8,8)
            dq = dequantization(block_8x8, quality)
            rec_block = idctn(dq)
            i = idx // W_segs 
            j = idx % W_segs 
            out[8*i:8*(i+1), 8*j:8*(j+1)] = rec_block
        return out[:H, :W]

    os.system("mkdir -p temp_frames_out")
    coeffs_per_channel = coeffs_per_frame // 3
    
    for i in range(num_frames):
        start_idx = i * coeffs_per_frame
        frame_chunk = decoded_values[start_idx : start_idx + coeffs_per_frame]
        y_c = frame_chunk[0 : coeffs_per_channel]
        cb_c = frame_chunk[coeffs_per_channel : 2*coeffs_per_channel]
        cr_c = frame_chunk[2*coeffs_per_channel : ]
        
        Y = rebuild_channel(y_c, (h, w))
        Cb = rebuild_channel(cb_c, (h, w))
        Cr = rebuild_channel(cr_c, (h, w))

        img = reassamble_image(Y, Cb, Cr)
        plt.imsave(f"temp_frames_out/f_{i:04d}.png", img.astype(np.uint8))
        print(f"\rSaved frame {i+1}/{num_frames}\033[K", end="", flush=True)

    if output_filename.endswith(".png"): output_filename = output_filename.replace(".png", ".mp4")
    os.system(f"ffmpeg -loglevel error -framerate 23.98 -i temp_frames_out/f_%04d.png -c:v libx264 -pix_fmt yuv420p {output_filename} -y")
    os.system("rm -r temp_frames_out/")
    print(f"\nDone: {output_filename}")

image_path = None
target_MSE = None
output_file_name = "output.png"
isVideo = False
do_serialize_file = False
do_deserialize_file = False

if len(sys.argv) > 1:
    image_path = sys.argv[1]
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        match arg:
            case "--error":
                target_MSE = int(sys.argv[i+1])
                i += 2
            case "-o":
                output_file_name = sys.argv[i+1]
                i += 2
            case "--Video":
                isVideo = True
                i += 1
            case "-si":
                do_serialize_file = True
                i += 1
            case "-so":
                do_deserialize_file = True
                i += 1
            case _:
                i += 1
else:
    print("Please give a file to compress!")
    exit()

if do_deserialize_file == True:
    if isVideo == True: 
        deserialize_video(image_path, output_file_name)
    else:
        deserialize_file(image_path, output_file_name)
    exit()

if isVideo == False:
    # Image Mode
    X = plt.imread(image_path)
    X = X[:, :, :3] # No Alpha
    if X.max() <= 1:
        X = X * 255
    X_c = None
    
    if target_MSE == None:
        if do_serialize_file == True:
            serialize_file(X, 1, output_file_name)
        else:
            X_c = jpeg_compress_image(X, 1)
            plt.imsave(output_file_name, X_c.astype(np.uint8))
    else:
        # Search for Target MSE
        MSE_v = 0
        i = 1
        jump_rate = 4
        while abs(MSE_v - target_MSE) > 3 and (jump_rate-1 > 10e-10):
            if MSE_v < target_MSE:
                i *= jump_rate 
            if MSE_v > target_MSE:
                jump_rate = np.sqrt(jump_rate)
                i /= jump_rate
            quality = 1/i
            X_c = jpeg_compress_image(X, quality)
            MSE_v = MSE(X,X_c)
            print(f"\rCurrent MSE: {MSE_v}\033[K", end="", flush=True)
        print()
        
        if do_serialize_file == True:
            serialize_file(X, 1/i, output_file_name)
        else:
            X_c = jpeg_compress_image(X, 1/i)
            plt.imsave(output_file_name, X_c.astype(np.uint8))
else:
    # Video Mode
    if do_serialize_file == True:
        if output_file_name == "output.png": output_file_name = "output.bin"
        serialize_video(image_path, 1/3, output_file_name)
    else: 
        # Frame-by-frame JPEG (No serialization)
        os.system("mkdir -p temp_frames")
        os.system(f"ffmpeg -loglevel error -i {image_path} temp_frames/f_%04d.png")

        frame_files = sorted(glob.glob("temp_frames/f_*.png"))
        for i,filepath in enumerate(frame_files):
            print(f"\rFrame #{i+1}", end="", flush=True)
            X = plt.imread(filepath)
            X = X[:, :, :3]
            if X.max() <=1 :
                X = X * 255
            X_c = jpeg_compress_image(X,1/3)
            plt.imsave(filepath, X_c.astype(np.uint8))
        print("\r\033[K")
        
        if output_file_name == "output.png":
            output_file_name = "output.mp4"

        # Merge video with Audio
        os.system(f"ffmpeg -loglevel error -framerate 23.98 -i temp_frames/f_%04d.png -i {image_path} -c:v libx264 -pix_fmt yuv420p -map 0:v:0 -map 1:a:0? {output_file_name} -y")
        os.system("rm -r temp_frames/")