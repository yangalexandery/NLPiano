import midi
import os

import numpy as np

LOWER_BOUND = 24
UPPER_BOUND = 102

SEGMENT_LENGTH = 128
DIVISION_LENGTH = 16


DIR_PREFIX = './data/'


def get_pieces():
    return [get_piece(DIR_PREFIX + midi_file) for midi_file in os.listdir(DIR_PREFIX)]

COUNT = 0


def get_piece(midi_file):
    global COUNT
    print(COUNT)
    COUNT += 1
    pattern = midi.read_midifile(midi_file)

    remaining_time = [track[0].tick for track in pattern]

    positions = [0 for _ in pattern]

    time = 0
    span = UPPER_BOUND - LOWER_BOUND

    state_matrix = []
    state = [[0, 0] for _ in xrange(span)]
    state_matrix.append(state)

    while True:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            old_state = state

            state = [[old_state[x][0], 0] for x in xrange(span)]
            state_matrix.append(state)

        for i in xrange(len(remaining_time)):
            while remaining_time[i] == 0:
                track = pattern[i]
                position = positions[i]

                event = track[position]
                if isinstance(event, midi.NoteEvent):
                    if (event.pitch < LOWER_BOUND) or (event.pitch >= UPPER_BOUND):
                        pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                    else:
                        if isinstance(event, midi.NoteOffEvent) or event.velocity == 0:
                            state[event.pitch - LOWER_BOUND] = [0, 0]
                        else:
                            state[event.pitch - LOWER_BOUND] = [1, 1]
                elif isinstance(event, midi.TimeSignatureEvent):
                    if event.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        return np.array([])

                try:
                    remaining_time[i] = track[position + 1].tick
                    positions[i] += 1
                except IndexError:
                    remaining_time[i] = None

            if remaining_time[i] is not None:
                remaining_time[i] -= 1

        if all(t is None for t in remaining_time):
            break

        time += 1
    print(np.array(state_matrix, dtype=np.float32).shape)
    return np.array(state_matrix, dtype=np.float32)

# def get_piece(midi_file):
#     pattern = midi.read_midifile(midi_file)

#     timeleft = [track[0].tick for track in pattern]

#     posns = [0 for track in pattern]

#     statematrix = []
#     span = UPPER_BOUND - LOWER_BOUND
#     time = 0

#     print(midi_file)
#     hasStarted = False

#     state = [[0, 0] for x in range(span)]
#     # statematrix.append(state)
#     print("Resolution: " + str(pattern.resolution))
#     starting_index = 0
#     while True:
#         if time % (pattern.resolution / 8) == (starting_index + 1) % (pattern.resolution / 8):#(pattern.resolution / 8):
#             # Crossed a note boundary. Create a new state, defaulting to holding notes
#             oldstate = state
#             state = [[oldstate[x][0], 0] for x in range(span)]
#             tot = sum([a[0] + a[1] for a in state])

#         for i in range(len(timeleft)):
#             while timeleft[i] == 0:
#                 track = pattern[i]
#                 pos = posns[i]

#                 evt = track[pos]
#                 if isinstance(evt, midi.NoteEvent):
#                     if (evt.pitch < LOWER_BOUND) or (evt.pitch >= UPPER_BOUND):
#                         pass
#                         print("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
#                     else:
#                         if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
#                             state[evt.pitch - LOWER_BOUND] = [0, 0]
#                         else:
#                             if not hasStarted:
#                                 starting_index = time
#                                 hasStarted = True
#                                 print(str(time) + "                                                      ALERT")
#                             state[evt.pitch - LOWER_BOUND] = [1, 1]
#                 elif isinstance(evt, midi.TimeSignatureEvent):
#                     if evt.numerator not in (2, 4):
#                         # We don't want to worry about non-4 time signatures. Bail early!
#                         # print "Found time signature event {}. Bailing!".format(evt)
#                         print('Found time signature event %s/%s. Bailing!' % (str(evt.numerator), str(evt.denominator)))
#                         return []

#                 try:
#                     timeleft[i] = track[pos + 1].tick
#                     posns[i] += 1
#                 except IndexError:
#                     timeleft[i] = None

#             if timeleft[i] is not None:
#                 timeleft[i] -= 1


#         if time % (pattern.resolution / 8) == starting_index % (pattern.resolution / 8): #(pattern.resolution / 8):
#             # Crossed a note boundary. Create a new state, defaulting to holding notes
#             tot = sum([a[0] + a[1] for a in state])

#             # code to remove leading all-0 states
#             if hasStarted:
#                 statematrix.append(state)

#         if all(t is None for t in timeleft):
#             break

#         time += 1

#     print(len(statematrix))
#     for i in range(len(statematrix)-1, -1, -1):
#         tot = sum([x[0] + x[1] for x in statematrix[i]]);
#         if tot > 0:
#             print(str(tot) + " " + str(i));
#             return statematrix[:i+1];
#     return np.array(statematrix, dtype=np.float32)

def print_statematrix(statematrix):
    count = 0
    for state in statematrix:
        s1 = ''
        s2 = ''
        for x in state:
            if x[0]==1 and x[1]==1:
                s1 += 'X'
            elif x[0] == 1 and x[1] == 0:
                s1 += '|'
            else:
                s1 += '-'
            # s1 += str(x[0])
            # s2 += str(x[1])
        if count % 4 == 0:
            s1 += ' --'
        count += 1
        print(s1)
        # print(s2)

def print_predictions(pred_matrix):
    for pred in pred_matrix[0]:
        s1 = ''
        s2 = ''
        for x in pred:
            a = np.argmax(x[0])
            b = np.argmax(x[1])
            # print(a, " ", b)
            if a == 1 and b == 1:
                s1 += 'X'
            elif a == 1 and b == 0:
                s1 += '|'
            else:
                s1 += '-'
        print(s1)


def print_predictions_2(pred_matrix):
    print(pred_matrix.shape)
    for pred in pred_matrix[0]:
        s1 = ''
        for x in pred:
            a = np.argmax(np.array([x[0], x[1], x[2]]))
            if a == 2:
                s1 += 'X'
            elif a == 1:
                s1 += '|'
            else:
                s1 += '-'
        print(s1)


if __name__ == '__main__':
    midi_filenames = sorted(os.listdir('./data'))
    print(midi_filenames[30])
    asdf = (get_piece('./data/' + midi_filenames[30]))
    print(len(asdf[0]))
    print(print_statematrix(asdf[:30]))
    # print(sum(asdf))
    # print(asdf)