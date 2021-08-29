import numpy as np
import glob
import datetime
import math
import random
import os
import shutil
import matplotlib.pyplot as plt
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track
import librosa.display
from tf2_utils import *

ROOT_PATH = '/Users/mckin/Desktop/cycleGANproj'
test_ratio = 0.1
LAST_BAR_MODE = 'remove'


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) != 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track

# run this with the path to your training dir, it copies files to the test dir, then only do #2 and down for each
"""1. divide the original set into train and test sets"""
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/classical/classical_train/origin_midi'))]
# print(l)
# idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
# print(len(idx))
# for i in idx:
#     shutil.move(os.path.join(ROOT_PATH, 'MIDI/classical/classical_train/origin_midi', l[i]),
#                 os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/origin_midi', l[i]))

"""2. convert_clean.py"""

"""3. choose the clean midi from original sets"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi'))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner'))]
# print(l)
# print(len(l))
# for i in l:
#     if os.path.exists(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/origin_midi', os.path.splitext(i)[0] + '.mid')):
#         shutil.copy(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/origin_midi', os.path.splitext(i)[0] + '.mid'),
#                     os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi', os.path.splitext(i)[0] + '.mid'))
#     else:
#         shutil.copy(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/origin_midi', os.path.splitext(i)[0] + '.midi'),
#                     os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi', os.path.splitext(i)[0] + '.midi'))

"""4. merge and crop"""
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi_gen')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi_gen'))
# if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_npy')):
#     os.makedirs(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_npy'))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi'))]
# print(l)
# count = 0
# for i in range(len(l)):
    
#     try:
    
#         multitrack = Multitrack(resolution=4, name=os.path.splitext(l[i])[0])
#         x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi', l[i]))
#         #multitrack.parse_pretty_midi(x)
#         multitrack = pypianoroll.from_pretty_midi(x)

#         '''
#         category_list = {'Piano': [], 'Drums': []}
#         program_dict = {'Piano': 0, 'Drums': 0}

#         for idx, track in enumerate(multitrack.tracks):
#             if track.is_drum:
#                 category_list['Drums'].append(idx)
#             else:
#                 category_list['Piano'].append(idx)
#         '''
#         non_percussion = []
#         for idx, track in enumerate(multitrack.tracks):
#             if (not track.is_drum):
#                 non_percussion.append(track)
        
#         to_merge = Multitrack(tracks=non_percussion, tempo=multitrack.tempo, downbeat=multitrack.downbeat, resolution=multitrack.resolution, name=multitrack.name)

#         tracks = []
#         #merged = multitrack[category_list['Piano']].get_merged_pianoroll()
#         merged = to_merge.blend()
#         print(merged.shape)

#         pr = get_bar_piano_roll(merged)
#         print(pr.shape)
#         pr_clip = pr[:, :, 24:108]
#         print(pr_clip.shape)
#         if int(pr_clip.shape[0] % 4) != 0:
#             pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
#         pr_re = pr_clip.reshape(-1, 64, 84, 1)
#         print(pr_re.shape)
#         save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_midi_gen', os.path.splitext(l[i])[0] +
#                                         '.mid'))
#         np.save(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
    
#     except:
#         count += 1
#         print('Wrong', l[i])
#         continue
    
# print(count)

"""5. concatenate into a big binary numpy array file"""
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_npy'))]
# print(l)
# train = np.load(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_npy', l[0]))
# print(train.shape, np.max(train))
# for i in range(1, len(l)):
#     print(i, l[i])
#     t = np.load(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/cleaner_npy', l[i]))
#     train = np.concatenate((train, t), axis=0)
# print(train.shape)
# np.save(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/classical_test_piano.npy'), (train > 0.0))

"""6. separate numpy array file into single phrases"""
if not os.path.exists(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/phrase_test')):
    os.makedirs(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/phrase_test'))
x = np.load(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/classical_test_piano.npy'))
print(x.shape)
count = 0
for i in range(x.shape[0]):
    if np.max(x[i]):
        count += 1
        np.save(os.path.join(ROOT_PATH, 'MIDI/classical/classical_test/phrase_test/classical_piano_test_{}.npy'.format(i+1)), x[i])
        print(x[i].shape)
   # if count == 11216:
   #     break
print(count)