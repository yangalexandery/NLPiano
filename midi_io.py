import midi
import os

LOWER_BOUND = 24
UPPER_BOUND = 102

SEGMENT_LENGTH = 128
DIVISION_LENGTH = 16


DIR_PREFIX = './data/'


def get_pieces():
    return [get_piece(DIR_PREFIX + midi_file) for midi_file in os.listdir(DIR_PREFIX)[0:20]]

def get_piece(midi_file):
    pattern = midi.read_midifile(midi_file)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    span = UPPER_BOUND - LOWER_BOUND
    time = 0

    print(midi_file)
    hasStarted = False

    state = [[0, 0] for x in range(span)]
    # statematrix.append(state)
    print("Resolution: " + str(pattern.resolution))
    starting_index = 0
    while True:
        if time % (pattern.resolution / 8) == (starting_index + 1) % (pattern.resolution / 8):#(pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            tot = sum([a[0] + a[1] for a in state])

        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < LOWER_BOUND) or (evt.pitch >= UPPER_BOUND):
                        pass
                        print("Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time))
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - LOWER_BOUND] = [0, 0]
                        else:
                            if not hasStarted:
                                starting_index = time
                                hasStarted = True
                                print(str(time) + "                                                      ALERT")
                            state[evt.pitch - LOWER_BOUND] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # We don't want to worry about non-4 time signatures. Bail early!
                        # print "Found time signature event {}. Bailing!".format(evt)
                        print('Found time signature event %s/%s. Bailing!' % (str(evt.numerator), str(evt.denominator)))
                        return []

                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1


        if time % (pattern.resolution / 8) == starting_index % (pattern.resolution / 8): #(pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            tot = sum([a[0] + a[1] for a in state])

            # code to remove leading all-0 states
            if hasStarted:
                statematrix.append(state)

        if all(t is None for t in timeleft):
            break

        time += 1

    print(len(statematrix))
    return statematrix

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

if __name__ == '__main__':
    midi_filenames = sorted(os.listdir('./data'))
    print(midi_filenames[30])
    asdf = (get_piece('./data/' + midi_filenames[30]))
    print(len(asdf[0]))
    print(print_statematrix(asdf[:30]))
    # print(sum(asdf))
    # print(asdf)