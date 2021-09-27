# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Download Flickr-Faces-HQ (FFHQ) dataset to current working directory."""

import os
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------

json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

tfrecords_specs = [
    dict(file_url='https://drive.google.com/uc?id=1LnhoytWihRRJ7CfhLQ76F8YxwxRDlZN3', file_path='tfrecords/ffhq/ffhq-r02.tfrecords', file_size=6860000,      file_md5='63e062160f1ef9079d4f51206a95ba39'),
    dict(file_url='https://drive.google.com/uc?id=1LWeKZGZ_x2rNlTenqsaTk8s7Cpadzjbh', file_path='tfrecords/ffhq/ffhq-r03.tfrecords', file_size=17290000,     file_md5='54fb32a11ebaf1b86807cc0446dd4ec5'),
    dict(file_url='https://drive.google.com/uc?id=1Lr7Tiufr1Za85HQ18yg3XnJXstiI2BAC', file_path='tfrecords/ffhq/ffhq-r04.tfrecords', file_size=57610000,     file_md5='7164cc5531f6828bf9c578bdc3320e49'),
    dict(file_url='https://drive.google.com/uc?id=1LnyiayZ-XJFtatxGFgYePcs9bdxuIJO_', file_path='tfrecords/ffhq/ffhq-r05.tfrecords', file_size=218890000,    file_md5='050cc7e5fd07a1508eaa2558dafbd9ed'),
    dict(file_url='https://drive.google.com/uc?id=1Lt6UP201zHnpH8zLNcKyCIkbC-aMb5V_', file_path='tfrecords/ffhq/ffhq-r06.tfrecords', file_size=864010000,    file_md5='90bedc9cc07007cd66615b2b1255aab8'),
    dict(file_url='https://drive.google.com/uc?id=1LwOP25fJ4xN56YpNCKJZM-3mSMauTxeb', file_path='tfrecords/ffhq/ffhq-r07.tfrecords', file_size=3444980000,   file_md5='bff839e0dda771732495541b1aff7047'),
    dict(file_url='https://drive.google.com/uc?id=1LxxgVBHWgyN8jzf8bQssgVOrTLE8Gv2v', file_path='tfrecords/ffhq/ffhq-r08.tfrecords', file_size=13766900000,  file_md5='74de4f07dc7bfb07c0ad4471fdac5e67'),
    dict(file_url='https://drive.google.com/uc?id=1M-ulhD5h-J7sqSy5Y1njUY_80LPcrv3V', file_path='tfrecords/ffhq/ffhq-r09.tfrecords', file_size=55054580000,  file_md5='05355aa457a4bd72709f74a81841b46d'),
    dict(file_url='https://drive.google.com/uc?id=1M11BIdIpFCiapUqV658biPlaXsTRvYfM', file_path='tfrecords/ffhq/ffhq-r10.tfrecords', file_size=220205650000, file_md5='bf43cab9609ab2a27892fb6c2415c11b'),
]

license_specs = {
    'json':      dict(file_url='https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX', file_path='LICENSE.txt',                    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'images':    dict(file_url='https://drive.google.com/uc?id=1sP2qz8TzLkzG2gjwAa4chtdB31THska4', file_path='images1024x1024/LICENSE.txt',    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'thumbs':    dict(file_url='https://drive.google.com/uc?id=1iaL1S381LS10VVtqu-b2WfF9TiY75Kmj', file_path='thumbnails128x128/LICENSE.txt',  file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'wilds':     dict(file_url='https://drive.google.com/uc?id=1rsfFOEQvkd6_Z547qhpq5LhDl2McJEzw', file_path='in-the-wild-images/LICENSE.txt', file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'tfrecords': dict(file_url='https://drive.google.com/uc?id=1SYUmqKdLoTYq-kqsnPsniLScMhspvl5v', file_path='tfrecords/ffhq/LICENSE.txt',     file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
}

#----------------------------------------------------------------------------

def download_file(session, file_spec, stats, chunk_size=128, num_attempts=10, **kwargs):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        with stats['lock']:
                            stats['bytes_done'] += len(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            if 'pixel_size' in file_spec or 'pixel_md5' in file_spec:
                with PIL.Image.open(tmp_path) as image:
                    if 'pixel_size' in file_spec and list(image.size) != file_spec['pixel_size']:
                        raise IOError('Incorrect pixel size', file_path)
                    if 'pixel_md5' in file_spec and hashlib.md5(np.array(image)).hexdigest() != file_spec['pixel_md5']:
                        raise IOError('Incorrect pixel MD5', file_path)
            break

        except:
            with stats['lock']:
                stats['bytes_done'] -= data_size

            # Handle known failure cases.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                data_str = data.decode('utf-8')

                # Google Drive virus checker nag.
                links = [html.unescape(link) for link in data_str.split('"') if 'export=download' in link]
                if len(links) == 1:
                    if attempts_left:
                        file_url = requests.compat.urljoin(file_url, links[0])
                        continue

                # Google Drive quota exceeded.
                if 'Google Drive - Quota exceeded' in data_str:
                    if not attempts_left:
                        raise IOError("Google Drive download quota exceeded -- please try again later")

            # Last attempt => raise error.
            if not attempts_left:
                raise

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic
    with stats['lock']:
        stats['files_done'] += 1

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

#----------------------------------------------------------------------------

def choose_bytes_unit(num_bytes):
    b = int(np.rint(num_bytes))
    if b < (100 << 0): return 'B', (1 << 0)
    if b < (100 << 10): return 'kB', (1 << 10)
    if b < (100 << 20): return 'MB', (1 << 20)
    if b < (100 << 30): return 'GB', (1 << 30)
    return 'TB', (1 << 40)

#----------------------------------------------------------------------------

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60: return '%ds' % s
    if s < 60 * 60: return '%dm %02ds' % (s // 60, s % 60)
    if s < 24 * 60 * 60: return '%dh %02dm' % (s // (60 * 60), (s // 60) % 60)
    if s < 100 * 24 * 60 * 60: return '%dd %02dh' % (s // (24 * 60 * 60), (s // (60 * 60)) % 24)
    return '>100d'

#----------------------------------------------------------------------------

def download_files(file_specs, num_threads=32, status_delay=0.2, timing_window=50, **download_kwargs):

    # Determine which files to download.
    done_specs = {spec['file_path']: spec for spec in file_specs if os.path.isfile(spec['file_path'])}
    missing_specs = [spec for spec in file_specs if spec['file_path'] not in done_specs]
    files_total = len(file_specs)
    bytes_total = sum(spec['file_size'] for spec in file_specs)
    stats = dict(files_done=len(done_specs), bytes_done=sum(spec['file_size'] for spec in done_specs.values()), lock=threading.Lock())
    if len(done_specs) == files_total:
        print('All files already downloaded -- skipping.')
        return

    # Launch worker threads.
    spec_queue = queue.Queue()
    exception_queue = queue.Queue()
    for spec in missing_specs:
        spec_queue.put(spec)
    thread_kwargs = dict(spec_queue=spec_queue, exception_queue=exception_queue, stats=stats, download_kwargs=download_kwargs)
    for _thread_idx in range(min(num_threads, len(missing_specs))):
        threading.Thread(target=_download_thread, kwargs=thread_kwargs, daemon=True).start()

    # Monitor status until done.
    bytes_unit, bytes_div = choose_bytes_unit(bytes_total)
    spinner = '/-\\|'
    timing = []
    while True:
        with stats['lock']:
            files_done = stats['files_done']
            bytes_done = stats['bytes_done']
        spinner = spinner[1:] + spinner[:1]
        timing = timing[max(len(timing) - timing_window + 1, 0):] + [(time.time(), bytes_done)]
        bandwidth = max((timing[-1][1] - timing[0][1]) / max(timing[-1][0] - timing[0][0], 1e-8), 0)
        bandwidth_unit, bandwidth_div = choose_bytes_unit(bandwidth)
        eta = format_time((bytes_total - bytes_done) / max(bandwidth, 1))

        print('\r%s %6.2f%% done  %d/%d files  %-13s  %-10s  ETA: %-7s ' % (
            spinner[0],
            bytes_done / bytes_total * 100,
            files_done, files_total,
            '%.2f/%.2f %s' % (bytes_done / bytes_div, bytes_total / bytes_div, bytes_unit),
            '%.2f %s/s' % (bandwidth / bandwidth_div, bandwidth_unit),
            'done' if bytes_total == bytes_done else '...' if len(timing) < timing_window or bandwidth == 0 else eta,
        ), end='', flush=True)

        if files_done == files_total:
            print()
            break

        try:
            exc_info = exception_queue.get(timeout=status_delay)
            raise exc_info[1].with_traceback(exc_info[2])
        except queue.Empty:
            pass

def _download_thread(spec_queue, exception_queue, stats, download_kwargs):
    with requests.Session() as session:
        while not spec_queue.empty():
            spec = spec_queue.get()
            try:
                download_file(session, spec, stats, **download_kwargs)
            except:
                exception_queue.put(sys.exc_info())

#----------------------------------------------------------------------------

def print_statistics(json_data):
    categories = defaultdict(int)
    licenses = defaultdict(int)
    countries = defaultdict(int)
    for item in json_data.values():
        categories[item['category']] += 1
        licenses[item['metadata']['license']] += 1
        country = item['metadata']['country']
        countries[country if country else '<Unknown>'] += 1

    for name in [name for name, num in countries.items() if num / len(json_data) < 1e-3]:
        countries['<Other>'] += countries.pop(name)

    rows = [[]] * 2
    rows += [['Category', 'Images', '% of all']]
    rows += [['---'] * 3]
    for name, num in sorted(categories.items(), key=lambda x: -x[1]):
        rows += [[name, '%d' % num, '%.2f' % (100.0 * num / len(json_data))]]

    rows += [[]] * 2
    rows += [['License', 'Images', '% of all']]
    rows += [['---'] * 3]
    for name, num in sorted(licenses.items(), key=lambda x: -x[1]):
        rows += [[name, '%d' % num, '%.2f' % (100.0 * num / len(json_data))]]

    rows += [[]] * 2
    rows += [['Country', 'Images', '% of all', '% of known']]
    rows += [['---'] * 4]
    for name, num in sorted(countries.items(), key=lambda x: -x[1] if x[0] != '<Other>' else 0):
        rows += [[name, '%d' % num, '%.2f' % (100.0 * num / len(json_data)),
            '%.2f' % (0 if name == '<Unknown>' else 100.0 * num / (len(json_data) - countries['<Unknown>']))]]

    rows += [[]] * 2
    widths = [max(len(cell) for cell in column if cell is not None) for column in itertools.zip_longest(*rows)]
    for row in rows:
        print("  ".join(cell + " " * (width - len(cell)) for cell, width in zip(row, widths)))

#----------------------------------------------------------------------------

def recreate_aligned_images(json_data, source_dir, dst_dir='realign1024x1024', output_size=1024, transform_size=4096, enable_padding=True, rotate_level=True, random_shift=0.0, retry_crops=False):
    print('Recreating aligned images...')

    # Fix random seed for reproducibility
    np.random.seed(12345)
    # The following random numbers are unused in present implementation, but we consume them for reproducibility
    _ = np.random.normal(0, 1, (len(json_data.values()), 2))

    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile('LICENSE.txt', os.path.join(dst_dir, 'LICENSE.txt'))

    for item_idx, item in enumerate(json_data.values()):
        print('\r%d / %d ... ' % (item_idx, len(json_data)), end='', flush=True)

        # Parse landmarks.
        # pylint: disable=unused-variable
        lm = np.array(item['in_the_wild']['face_landmarks'])
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        if rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1

        # Load in-the-wild image.
        src_file = os.path.join(source_dir, item['in_the_wild']['file_path'])
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
        qsize = np.hypot(*x) * 2

        # Keep drawing new random crop offsets until we find one that is contained in the image
        # and does not require padding
        if random_shift != 0:
            for _ in range(1000):
                # Offset the crop rectange center by a random shift proportional to image dimension
                # and the requested standard deviation
                c = (c0 + np.hypot(*x)*2 * random_shift * np.random.normal(0, 1, c0.shape))
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
                if not retry_crops or not (crop[0] < 0 or crop[1] < 0 or crop[2] >= img.width or crop[3] >= img.height):
                    # We're happy with this crop (either it fits within the image, or retries are disabled)
                    break
            else:
                # rejected N times, give up and move to next image
                # (does not happen in practice with the FFHQ data)
                print('rejected image')
                return

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        dst_subdir = os.path.join(dst_dir, '%05d' % (item_idx - item_idx % 1000))
        os.makedirs(dst_subdir, exist_ok=True)
        img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))

    # All done.
    print('\r%d / %d ... done' % (len(json_data), len(json_data)))

#----------------------------------------------------------------------------

def run(tasks, **download_kwargs):
    if not os.path.isfile(json_spec['file_path']) or not os.path.isfile('LICENSE.txt'):
        print('Downloading JSON metadata...')
        download_files([json_spec, license_specs['json']], **download_kwargs)

    print('Parsing JSON metadata...')
    with open(json_spec['file_path'], 'rb') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    if 'stats' in tasks:
        print_statistics(json_data)

    specs = []
    if 'images' in tasks:
        specs += [item['image'] for item in json_data.values()] + [license_specs['images']]
    if 'thumbs' in tasks:
        specs += [item['thumbnail'] for item in json_data.values()] + [license_specs['thumbs']]
    if 'wilds' in tasks:
        specs += [item['in_the_wild'] for item in json_data.values()] + [license_specs['wilds']]
    if 'tfrecords' in tasks:
        specs += tfrecords_specs + [license_specs['tfrecords']]

    if len(specs):
        print('Downloading %d files...' % len(specs))
        np.random.shuffle(specs) # to make the workload more homogeneous
        download_files(specs, **download_kwargs)

    if 'align' in tasks:
        recreate_aligned_images(json_data, source_dir=download_kwargs['source_dir'], rotate_level=not download_kwargs['no_rotation'], random_shift=download_kwargs['random_shift'], enable_padding=not download_kwargs['no_padding'], retry_crops=download_kwargs['retry_crops'])

#----------------------------------------------------------------------------

def run_cmdline(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Download Flickr-Face-HQ (FFHQ) dataset to current working directory.')
    parser.add_argument('-j', '--json',         help='download metadata as JSON (254 MB)', dest='tasks', action='append_const', const='json')
    parser.add_argument('-s', '--stats',        help='print statistics about the dataset', dest='tasks', action='append_const', const='stats')
    parser.add_argument('-i', '--images',       help='download 1024x1024 images as PNG (89.1 GB)', dest='tasks', action='append_const', const='images')
    parser.add_argument('-t', '--thumbs',       help='download 128x128 thumbnails as PNG (1.95 GB)', dest='tasks', action='append_const', const='thumbs')
    parser.add_argument('-w', '--wilds',        help='download in-the-wild images as PNG (955 GB)', dest='tasks', action='append_const', const='wilds')
    parser.add_argument('-r', '--tfrecords',    help='download multi-resolution TFRecords (273 GB)', dest='tasks', action='append_const', const='tfrecords')
    parser.add_argument('-a', '--align',        help='recreate 1024x1024 images from in-the-wild images', dest='tasks', action='append_const', const='align')
    parser.add_argument('--num_threads',        help='number of concurrent download threads (default: 32)', type=int, default=32, metavar='NUM')
    parser.add_argument('--status_delay',       help='time between download status prints (default: 0.2)', type=float, default=0.2, metavar='SEC')
    parser.add_argument('--timing_window',      help='samples for estimating download eta (default: 50)', type=int, default=50, metavar='LEN')
    parser.add_argument('--chunk_size',         help='chunk size for each download thread (default: 128)', type=int, default=128, metavar='KB')
    parser.add_argument('--num_attempts',       help='number of download attempts per file (default: 10)', type=int, default=10, metavar='NUM')
    parser.add_argument('--random-shift',       help='standard deviation of random crop rectangle jitter', type=float, default=0.0, metavar='SHIFT')
    parser.add_argument('--retry-crops',        help='retry random shift if crop rectangle falls outside image (up to 1000 times)', dest='retry_crops', default=False, action='store_true')
    parser.add_argument('--no-rotation',        help='keep the original orientation of images', dest='no_rotation', default=False, action='store_true')
    parser.add_argument('--no-padding',         help='do not apply blur-padding outside and near the image borders', dest='no_padding', default=False, action='store_true')
    parser.add_argument('--source-dir',         help='where to find already downloaded FFHQ source data', default='', metavar='DIR')

    args = parser.parse_args()
    if not args.tasks:
        print('No tasks specified. Please see "-h" for help.')
        exit(1)
    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_cmdline(sys.argv)

#----------------------------------------------------------------------------
