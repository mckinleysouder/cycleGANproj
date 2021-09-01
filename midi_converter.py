from os import listdir
from os.path import isfile, join
import pretty_midi
import pypianoroll
from pypianoroll import Multitrack, Track, StandardTrack
import numpy as np

# Combine tracks and remove percussion, returns and takes in a multitrack
def get_merged(multitrack):
    non_percussion = []
    for idx, track in enumerate(multitrack.tracks):
        if (not track.is_drum):
            non_percussion.append(track)

    to_merge = Multitrack(tracks=non_percussion, tempo=multitrack.tempo, downbeat=multitrack.downbeat, resolution=multitrack.resolution, name=multitrack.name)
    blended = to_merge.blend()
    # return blended
    blendedTrack = Track(pianoroll=blended, program=0, name="BlendedTrack")
    return Multitrack(tracks=[blendedTrack], tempo=multitrack.tempo, downbeat=multitrack.downbeat, resolution=multitrack.resolution, name=multitrack.name)

# to know if data is good
def midi_filter(midi_info):
    """Return True for qualified midi files and False for unwanted ones"""
    if midi_info['first_beat_time'] > 0.0:
        return False
    elif midi_info['num_time_signature_change'] > 1:
        return False
    elif midi_info['time_signature'] not in ['4/4']:
        return False
    return True

# says if data is good or bad
def skip_bad_data(pm):
    if pm.time_signature_changes:
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time
    else:
        first_beat_time = pm.estimate_beat_start()
    tc_times, tempi = pm.get_tempo_changes()
    if len(pm.time_signature_changes) == 1:
        time_sign = '{}/{}'.format(pm.time_signature_changes[0].numerator,
                                   pm.time_signature_changes[0].denominator)
    else:
        time_sign = None
    midi_info = {
        'first_beat_time': first_beat_time,
        'num_time_signature_change': len(pm.time_signature_changes),
        'time_signature': time_sign,
        'tempo': tempi[0] if len(tc_times) == 1 else None
    }
    if not midi_filter(midi_info):
        print("Bad Data")
        return 1
    else:
        print("Good Data")
        return 0

def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) != 0:
        piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll

# START -----------------------------------------------------------------------------------------------

# Get Preprocssed data and store in on preprocessed_files
preprocessed_path = 'Blues/Preprocessed/'
preprocessed_files = [f for f in listdir(preprocessed_path) if isfile(join(preprocessed_path, f))]

# iterate through each MIDI file up end
to_convert = 0
end = 5
for file_name in preprocessed_files:
    if to_convert >= end:
        break

    # ignore bad data
    pm = pretty_midi.PrettyMIDI(preprocessed_path+file_name, file_name)
    if skip_bad_data(pm):
        continue
    
    # load in data
    multitrack = Multitrack(resolution=4, name=file_name)
    x = pretty_midi.PrettyMIDI('Blues/Preprocessed/'+file_name, file_name)
    multitrack = pypianoroll.from_pretty_midi(x)

    # remove drums
    non_percussion = []
    for idx, track in enumerate(multitrack.tracks):
        if (not track.is_drum):
            non_percussion.append(track)
    
    # merge tracks
    to_merge = Multitrack(tracks=non_percussion, tempo=multitrack.tempo, downbeat=multitrack.downbeat, resolution=multitrack.resolution, name=multitrack.name)
    tracks = []

    # reshape
    merged = to_merge.blend()
    print(merged.shape)

    sample_number = 0
    while sample_number+64 < merged.shape[0]:
        # np.save(file='/Blues/Postprocessed/Blues_sample_'+str(sample_number/64)+'.npy', arr=merged[sample_number:sample_number+64, :])
        #print(merged[sample_number:sample_number+64, :])
        Multitrack(tracks=[StandardTrack(pianoroll=merged[sample_number:sample_number+64, :])]).to_pretty_midi().write('/Blues/Postprocessed/Blues_sample_'+str(sample_number/64)+'.midi')
        sample_number+=64
    
    to_convert+=1

