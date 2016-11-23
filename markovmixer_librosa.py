#!/usr/bin/env python2


####################################################
#                                                  #
#                  Markov Mixer                    #
#                    EE 126                        #
#                                                  #
####################################################


from __future__ import division
from collections import defaultdict

import os, os.path
import sys
import argparse
import cPickle, pickle
import random
import bisect
#import shelve # you may want to look into this, or a database if you want
#              # to use a decently sized set of songs

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import scikits.samplerate as samplerate
import pyaudio
import librosa



TARGET_FS = 44100
NUM_CHANNELS = 1



def load_song(song_path):
    """
    Loads an individual song, normalizes it, and returns it.
    """
    fs, data = wavfile.read(song_path)
    data = data.mean(axis=1).flatten()
    data = samplerate.resample(data, TARGET_FS/fs, 'linear')

    # Normalize
    data = data/np.max(data)*.95
    print "\t- Loaded %s" % song_path
    return data



def load_songs(song_dir, max_songs=10, valid_exts=('.wav')):
    """
    Loads the songs in a given directory, and returns a list of song names,
    and a dictionary mapping from song titles to the audio signal.
    """
    song_paths = ["%s/%s" % (song_dir, s) for s in os.listdir(song_dir) \
                    if os.path.splitext(s)[1] in valid_exts]

    if max_songs == -1:
        max_songs = len(song_paths)

    song_paths = song_paths[:min(max_songs, len(song_paths))]
    song_names = [os.path.splitext(s.split("/")[-1])[0] for s in song_paths]

    song_name2wav = dict(zip(song_names, [load_song(s) for s in song_paths]))

    return song_names, song_name2wav



def compute_bisong_xcorr_peaks(s1_name, s1_wav, s1_beats, s2_name, s2_wav, \
        s2_beats, xcorr_thresh):
    """
    Computes the (magnitude) cross-correlation of the two provided songs at
    time windows corresponding to beats. We accumulate the cross-correlation
    values that exceed the threshold and return them.
    """
    #
    #   If you want to play around with the cross-correlation, introduce
    #   a different similarity metric, look here, and in
    #   'compute_all_xcorr_peaks'
    #


    s1_len = s1_wav.size
    s2_len = s2_wav.size

    s1_xcorrs = []
    s2_xcorrs = []

    # Calculate sample index pairs for the highly correlated segments
    for i in xrange(len(s1_beats)):
        s1_start = s1_beats[i]
        s1_end = s1_wav.size if i == len(s1_beats)-1 else s1_beats[i+1]

        for j in xrange(len(s2_beats)):
            s2_start = s2_beats[j]
            s2_end = s2_wav.size if j == len(s2_beats)-1 else s2_beats[j+1]

            # If the segments don't overlap, choose the smaller width
            seg_len = min(s1_end - s1_start, s2_end - s2_start)
            s2_end = s2_start + seg_len

            s1_seg = s1_wav[s1_start:s1_start+seg_len]
            s2_seg = s2_wav[s2_start:s2_end]

            # Calculate the "cross-correlation"
            xcorr = np.abs(np.dot(s1_seg, s2_seg))

            # Save this as a jump (transition) point, if it exceeds the
            # provided threshold
            if xcorr >= xcorr_thresh:
                s1_xcorrs.append((s2_name, s1_start, s2_start, xcorr))
                s2_xcorrs.append((s1_name, s2_start, s1_start, xcorr))

    print "\t\t- %d Matches" % len(s1_xcorrs)

    return s1_xcorrs, s2_xcorrs



def compute_all_xcorr_peaks(song_names, song_name2wav, s_beats, \
                            max_jumps_per_song, xcorr_thresh):
    """
    Computes the cross-correlation peaks between the beats of the song, which
    serve as our similarity metric. A dictionary that maps from song names to
    the cross-correlation peak information (the outgoing song, the indices of
    the transitions from either song to the other, and the cross
    correlation value).
    """
    #
    #   If you want to play around with the cross-correlation, introduce
    #   a different similarity metric, look here, and in
    #   'compute_bisong_xcorr_peaks'
    #


    # For each song, store a list of tuples:
    # (jump_song_name, start_idx, jump_idx, xcorr)
    song_name2xcorr = defaultdict(list)

    for i in xrange(0, len(song_names)-1):
        s1_name = song_names[i]
        s1_wav = song_name2wav[s1_name]
        s1_beats = s_beats[i]


        for j in xrange(i+1, len(song_names)):
            if i == j:
                continue
            print "\t- Finding xcorr for song pair %d-%d" % (i, j)
            s2_name = song_names[j]
            s2_wav = song_name2wav[s2_name]
            s2_beats = s_beats[j]
            s1_xcorrs, s2_xcorrs = compute_bisong_xcorr_peaks(s1_name, s1_wav, \
                    s1_beats, s2_name, s2_wav, s2_beats, xcorr_thresh)

            song_name2xcorr[s1_name].extend(s1_xcorrs)
            song_name2xcorr[s2_name].extend(s2_xcorrs)

        if len(song_name2xcorr[s1_name]) > max_jumps_per_song:
            # Pick the best peaks if we have too many

            max_end = min(max_jumps_per_song, len(song_name2xcorr[s1_name]))

            song_name2xcorr[s1_name] = sorted(song_name2xcorr[s1_name], \
                 key = lambda (n, sid, jid, xcor): -xcor)[:max_end]

    return song_name2xcorr



def prune_isolated_songs(song_names, song_name2wav, song_name2xcorr):
    """
    Remove songs that have no transitions.
    """
    removed_songs = [s_name for s_name in song_name2xcorr \
                     if s_name not in song_name2xcorr]
    for s_name in removed_songs:
        del song_name2wav[s_name]
        song_names.remove(s_name)



def compute_transition_probs(song_names, song_name2xcorr, max_jumps_per_song, \
                             spread, xcorr_shift, prob_scale):
    """
    Calculate transition probabilities derived from the cross-correlation
    values (our metric of similarity) between correlated samples between songs.
    """

    #
    #   If you want to play with the transition probabilities, you may want
    #   to start here.
    #

    song_name2jumptimes = defaultdict(list)
    song_name2jumps = defaultdict(list)

    # For each song, store a list of tuples:
    # (jump_song_name, start_idx, jump_idx, jump_prob)
    for s_name in song_names:
        start_idxs_to_jumps = defaultdict(list)

        # Remove songs that are not loaded
	song_name2xcorr[s_name] = \
	    filter(lambda (s2, s1_t, s2_t, xc): s2 in song_names, \
		   song_name2xcorr[s_name])

        # This is repeated here so that we can lower the number of transitions
        # without having to recompute the autocorrelations and beats again
        if len(song_name2xcorr[s_name]) > max_jumps_per_song:
            # Pick the best peaks if we have too many

            max_end = min(max_jumps_per_song, len(song_name2xcorr[s_name]))

            song_name2xcorr[s_name] = sorted(song_name2xcorr[s_name], \
                 key = lambda (n, sid, jid, xcor): -xcor)[:max_end]

        for s2_name, start_idx, jump_idx, xcorr in song_name2xcorr[s_name]:
            # Map cross-correlation to a probability using the logistic function
            jump_prob = prob_scale/(1 + np.exp(-(xcorr + xcorr_shift)/spread))

            jump = (s2_name, jump_idx, jump_prob)
            start_idxs_to_jumps[start_idx].append(jump)

        # Handle any jumps that may start from the same index, and put them
        # into a 'jump group'
        for start_idx, jump_group in start_idxs_to_jumps.items():
            num_jumps = len(jump_group)
            if num_jumps > 1:
                jumps = []
                # Reweight the transition probs in the same group equally
                # so they don't exceed 1
                for s2_name, jump_idx, jump_prob in jump_group:
                    jump = (s2_name, jump_idx, jump_prob/num_jumps)
                    jumps.append(jump)
                song_name2jumptimes[s_name].append(start_idx)
                song_name2jumps[s_name].append(jumps)
            else:
                song_name2jumptimes[s_name].append(start_idx)
                song_name2jumps[s_name].append(jump_group)

    return song_name2jumptimes, song_name2jumps



class MarkovMixer(object):
    """
    Generates an audio stream composed of segments from a corpus of songs
    that have a probability of transitioning between similar points in
    other songs.
    """

    def __init__(self, song_name2wav, song_name2jumptimes, \
            song_name2jumps, force_restart=False):
        self.song_names = song_name2jumptimes.keys()
        self.song_name2wav = song_name2wav
        self.song_name2jumptimes = song_name2jumptimes
        self.song_name2jumps = song_name2jumps
        self.force_restart = force_restart

        # Sort the songs in temporal order
        for song in self.song_name2jumps.keys():
            sorted_order = sorted(range(len(self.song_name2jumptimes[song])), \
                    key=self.song_name2jumptimes[song].__getitem__)
            self.song_name2jumptimes[song] = \
                    [self.song_name2jumptimes[song][i] for i in sorted_order]
            self.song_name2jumps[song] = \
                    [self.song_name2jumps[song][i] for i in sorted_order]

        # State information
        # Initialize with a random song
        self.curr_song_name = np.random.choice(self.song_names)
        self.curr_idx = 0

        # Only used to print the first song
        self.init_flag = True



    def get_jump_group_idxs_in_frame(self, samples_left):
        """
        Returns the first or last index of the 'jump group' that have
        corresponding transition times that are within the remaining
        time interval of the frame.
        """

        jumptimes = self.song_name2jumptimes[self.curr_song_name]
        left_idx = bisect.bisect_left(jumptimes, self.curr_idx)
        right_idx = bisect.bisect_left(jumptimes, self.curr_idx+samples_left)-1

        if left_idx == len(jumptimes):
            return -1, -1

        left_time = jumptimes[left_idx]
        right_time = jumptimes[right_idx]

        if left_time < self.curr_idx or left_time >= self.curr_idx+samples_left:
            return -1, -1

        assert jumptimes[left_idx] >= self.curr_idx
        assert jumptimes[right_idx] < self.curr_idx + samples_left, \
                "Jump Index %d: sample %d" % (right_idx, jumptimes[right_idx])

        return  left_idx, right_idx



    def handle_frame(self, in_data, sample_count, time_info, status):
        """
        Upon the request of pyAudio, returns the current frame of the
        randomly generated musical audio stream.
        """

        if self.init_flag:
            print "Starting song: %s" % self.curr_song_name
            self.init_flag = False

        frame = np.zeros(sample_count)
        potential_jumps = self.song_name2jumps[self.curr_song_name]

        frame = np.zeros(sample_count, dtype=float)

        # The index for the current frame
        idx = 0

        new_song = True

        # Check for possible jumps

        # Loop condition just as a safeguard against writing too many samples
        # to the frame
        while idx < sample_count - 1:

            # If we have transitioned to a new song, get the indices for the
            # jumps to other songs that occur during this frame
            if new_song:
                i, end = self.get_jump_group_idxs_in_frame(sample_count - idx)

                # If there were no valid jumps, then we have no transitions
                # left in this frame, so we stop looking
                if i == -1:
                    break

                new_song = False

            # Get a new jump
            start_idx = self.song_name2jumptimes[self.curr_song_name][i]
            jump_group = self.song_name2jumps[self.curr_song_name][i]

            # Obtain a sample for the transition
            cumul_prob = 0.0
            rand_samp = random.random()

            sys.stdout.write(">")

            # Iterate through all jumps at this time, and determine if the
            # sample results in a jump or not.
            for s2_song, jump_idx, jump_prob in jump_group:

                cumul_prob += jump_prob

                # If we successfully have a transition, fill the frame up to
                # 'idx' with samples from the previous song
                if rand_samp < cumul_prob:
                    last_dur = start_idx - self.curr_idx
                    last_wav = self.song_name2wav[self.curr_song_name]
                    frame[idx:idx+last_dur] = \
                            last_wav[self.curr_idx:self.curr_idx+last_dur]

                    # Advance the frame index
                    idx += last_dur

                    # Transition to the new song
                    self.curr_song_name = s2_song
                    self.curr_idx = jump_idx
                    print "\nTransitioned with p=%f to: %s" \
                            % (jump_prob, self.curr_song_name)

                    new_song = True
                    break

            # If we have finished going through the possible jumps in this
            # frame, break out of the loop
            if i == end:
                break


            # Go to the next jump time
            i += 1

        wav = self.song_name2wav[self.curr_song_name]
        dur = min(sample_count - idx, wav.size - self.curr_idx)

        # Fill the rest of the frame with samples from the current song
        frame[idx:idx+dur] = wav[self.curr_idx:self.curr_idx+dur]
        # Advance the index for the current song
        self.curr_idx += dur

        # If we reached the end of the song, then we are done. Otherwise,
        # we continue processing frames. However, if we have the force_restart
        # flag set, we randomly pick a new song and start from the beginning.
        if self.curr_idx == wav.size:
            if self.force_restart:
                self.curr_song_name = np.random.choice(self.song_names)
                self.curr_idx = 0
                if idx + dur < sample_count:
                    # We assume here that we have songs longer than a frame
                    wav = self.song_name2wav[self.curr_song_name]
                    frame[idx+dur:] = wav[:sample_count-idx-dur]
                    self.curr_idx += idx + dur

                    print "\nReached the end, restarting at: %s" \
                            % self.curr_song_name

                return_code = pyaudio.paContinue

            else:
                return_code = pyaudio.paComplete
        else:
            return_code = pyaudio.paContinue

        # Finally, return the current frame and the pyaudio return code
        return frame, return_code



def encode_audio(signal):
    """
    Convert a 2D numpy array into a byte sequence for PyAudio
    Signal should be a numpy array with shape (chunk_size, NUM_CHANNELS)
    """
    interleaved = signal.flatten()
    out_data = interleaved.astype(np.float32).tostring()
    return out_data



def make_markov_mixer_callback(mixer):
    """
    Makes a callback for pyaudio, which queries the given MarkovMixer for new
    audio frames.
    """

    def get_next_frame(in_data, frame_count, time_info, status):
        # Have the mixer provide an audio frame
        sample_frame, return_code = mixer.handle_frame(in_data, frame_count, \
                time_info, status)
        # Encode the audio in byte format
        char_frame = encode_audio(sample_frame)
        if return_code == pyaudio.paComplete:
            print "End reached. Please type 'exit' to exit."
        return (char_frame, return_code)

    return get_next_frame



def get_arguments():
    """
    Gets the command line arguments for this program.
    """
    parser = argparse.ArgumentParser(description="Markov Chain Song Mixer")

    parser.add_argument("song_dir", action="store", type=str, \
            help="The directory where the songs are located.")

    parser.add_argument("--num_songs", "-n", action="store", dest="max_songs",\
            type=int, default=-1, \
            help="The number of songs to be used, picked arbitrarily. If \
                  unspecified, all songs are used.")

    parser.add_argument("--beats", "-b", action="store", dest="bps",\
            type=int, default=4, \
            help="Sets the number of detected beats per segment.")

    parser.add_argument("--jumps", "-j", action="store", \
            dest="max_jumps_per_song", type=int, default=100, \
            help="Sets the maximum number of jumps per song. The default \
                  number is 100.")

    parser.add_argument("--spread", "-s", action="store", \
            dest="spread", type=float, default=1000, \
            help="This factor affects the spread of cross-correlation when " \
                 +"they are mapped to a probability via the logistic function.")

    parser.add_argument("--offset", "-o", action="store", \
            dest="xcorr_shift", type=float, default=10, \
            help="This factor shifts the center of the logistic used to map " \
                 + "cross-correlation to probabilities.")

    parser.add_argument("--probscale", "-p", action="store", \
            dest="prob_scale", type=float, default=1, \
            help="Linearly scales the transition probabilities. Used to " \
                + "tune the overall probability of staying in a state when " \
                + "transitions are possible.")

    parser.add_argument("--recompute", "-r", action="store_true", \
            dest="recompute", default=False, \
            help="When set, forces recomputation of beats and \
            cross-correlation values, regardless of if they have been saved.")

    parser.add_argument("--forcerestart", "-f", action="store_true", \
            dest="force_restart", default=False, \
            help="When set, forces the Markov Mixer to randomly start from a " \
                 + "new song after reaching the end, rather than terminating.")

    parser.add_argument("--threshold", "-t", action="store", \
            dest="xcorr_threshold", type=float, default=0.05, \
            help="The cross-correlation threshold with which we reject " \
                 + "transitions associated with insufficiently " \
                 + "cross-correlated segments.")

    args = parser.parse_args()

    return args



def extract_beats(song_names, song_name2wav, bps):
    """
    Finds and returns indices of beat-like events in the signal. Generally
    this would correspond to the rhythm you feel when listening to a song,
    but could be slightly different.
    """

    s_beats = []

    for i, s_name in enumerate(song_names):
        s_wav = song_name2wav[s_name]
        tempo, s_bs = librosa.beat.beat_track(s_wav, TARGET_FS, hop_length=128)

        # Librosa returns beats indexed by STFT frames, so convert to seconds
        # and then back to sample indices
        s_bs = (librosa.frames_to_time(s_bs, TARGET_FS, hop_length=128) \
                * TARGET_FS).astype(int)

        # Use every two beats instead of just one
        s_bs = s_bs[::bps]
        #s_bs = (s_bs*TARGET_FS).astype(int)
        s_beats.append(s_bs)

    return s_beats



def run_markov_mixer():
    args = get_arguments()

    song_dir = args.song_dir
    bps = args.bps
    max_songs = args.max_songs
    xcorr_threshold = args.xcorr_threshold
    spread = args.spread
    xcorr_shift = args.xcorr_shift
    recompute = args.recompute
    prob_scale = args.prob_scale
    max_jumps_per_song = args.max_jumps_per_song
    force_restart = args.force_restart

    # Validate args
    assert max_songs > 0 or max_songs == -1, \
        "Maximum number of songs must be positive!"
    assert bps > 0, "Must have a positive number of beats per segment!"
    assert max_jumps_per_song > 0, \
            "Maximum number of jumps per songs must be positive!"
    assert prob_scale >= 0 and prob_scale <= 1, \
            "The probability scaling factor must be between [0,1]!"


    print " * Loading songs..."
    song_names, song_name2wav = load_songs(song_dir, max_songs=max_songs)

    print " * Extracting beats..."


    if os.path.exists("%s/beat_idxs.pkl" % song_dir) and not recompute:
        with open("%s/beat_idxs.pkl" % song_dir, "rb") as f:
            s_beats = cPickle.load(f)
    else:
        s_beats = extract_beats(song_names, song_name2wav, bps)

        # Save the beats so we don't have to wait every time
        with open("%s/beat_idxs.pkl" % song_dir, "wb") as f:
            cPickle.dump(s_beats, f, pickle.HIGHEST_PROTOCOL)

    print " * Computing cross-correlations..."

    if os.path.exists("%s/xcorr.pkl" % song_dir) and not recompute:
        with open("%s/xcorr.pkl" % song_dir, "rb") as f:
            song_name2xcorr = cPickle.load(f)
    else:
        song_name2xcorr = compute_all_xcorr_peaks(song_names, \
                song_name2wav, s_beats, max_jumps_per_song, xcorr_threshold)

        # Save cross correlation so we don't have to wait every time
        with open("%s/xcorr.pkl" % song_dir, "wb") as f:
            cPickle.dump(song_name2xcorr, f, pickle.HIGHEST_PROTOCOL)

    print " * Pruning songs without transitions..."
    prune_isolated_songs(song_names, song_name2wav, song_name2xcorr)

    print " * Computing transition probabilities..."
    song_name2jumptimes, song_name2jumps = compute_transition_probs(song_names,\
            song_name2xcorr, max_jumps_per_song, spread, \
            xcorr_shift, prob_scale)

    print " * Setting up Markov Mixer..."
    mixer = MarkovMixer(song_name2wav, song_name2jumptimes,
                        song_name2jumps, force_restart)

    print " * Setting up pyAudio stream...\n"
    p = pyaudio.PyAudio()

    print "\n\t - NOTE: pyAudio may cause some ALSA warnings, but this is " \
              + "usually not a concern."

    stream = p.open(format=pyaudio.paFloat32,
                    channels=NUM_CHANNELS,
                    rate=TARGET_FS,
                    output=True,
                    stream_callback=make_markov_mixer_callback(mixer))

    print "\nDone. Let's get our mix on! :D When there is a potential, " \
            + "transition, you will see a '>' appear. When a transition is " \
            + "taken, the title of the new song will be announced.\n\n"

    # Input loop, currently only for exiting
    # You could add some interesting user interaction with the Markov chain.
    exit_flag = False
    while not exit_flag:
        exit_flag = (raw_input("Type 'exit' to exit.\n\n").lower() == "exit")

    stream.close()


if __name__ == "__main__":
    run_markov_mixer()
    exit(0)
