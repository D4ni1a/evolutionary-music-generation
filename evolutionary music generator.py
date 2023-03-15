import random
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage
import music21
import mingus.core.chords as chords
import itertools
from math import sqrt

"""Constants for running EA and output of melody"""
POPULATION_SIZE = 500
GENERATION = 300
CHILDREN = 120
OCTAVE = 4
BAR = 1536
HALF_BAR = int(BAR / 2)
QUARTER_BAR = int(BAR / 4)
INPUT_FILE = "input2.mid"
OUTPUT_FILE = "test.mid"

"""Class for parsing and preprocessing midi files' data"""
class MusicLibrary:
    """List of 12 notes and Minor/Major key table"""
    def __init__(self):
        self.minor_major_tables = {
            "minor": {"C": {"C": "minor", "D": "dim", "D#": "major", "F": "minor", "G": "minor", "G#": "major",
                            "A#": "major"},
                      "C#": {"C#": "minor", "D#": "dim", "E": "major", "F#": "minor", "G#": "minor", "A": "major",
                             "B": "major"},
                      "D": {"D": "minor", "E": "dim", "F": "major", "G": "minor", "A": "minor", "A#": "major",
                            "C": "major"},
                      "D#": {"D#": "minor", "E#": "dim", "F#": "major", "G#": "minor", "A#": "minor", "B": "major",
                             "C#": "major"},
                      "E": {"E": "minor", "F#": "dim", "G": "major", "A": "minor", "B": "minor", "C": "major",
                            "D": "major"},
                      "F": {"F": "minor", "G": "dim", "G#": "major", "A#": "minor", "C": "minor", "C#": "major",
                            "D#": "major"},
                      "F#": {"F#": ""
                                   "", "G#": "dim", "A": "major", "B": "minor", "C#": "minor", "D": "major",
                             "E": "major"},
                      "G": {"G": "minor", "A": "dim", "A#": "major", "C": "minor", "D": "minor", "D#": "major",
                            "F": "major"},
                      "G#": {"G#": "minor", "A#": "dim", "B": "major", "C#": "minor", "D#": "minor", "E": "major",
                             "F#": "major"},
                      "A": {"A": "minor", "B": "dim", "C": "major", "D": "minor", "E": "minor", "F": "major",
                            "G": "major"},
                      "A#": {"A#": "minor", "B#": "dim", "C#": "major", "D#": "minor", "E#": "minor", "F#": "major",
                             "G#": "major"},
                      "B": {"B": "minor", "C#": "dim", "D": "major", "E": "minor", "F#": "minor", "G": "major",
                            "A": "major"}},
            "major": {
                "C": {"C": "major", "D": "minor", "E": "minor", "F": "major", "G": "major", "A": "minor", "B": "dim"},
                "C#": {"C#": "major", "D#": "minor", "E#": "minor", "F#": "major", "G#": "major", "A#": "minor",
                       "B#": "dim"},
                "D": {"D": "major", "E": "minor", "F#": "minor", "G": "major", "A": "major", "B": "minor", "C#": "dim"},
                "D#": {"D#": "major", "F": "minor", "G": "minor", "G#": "major", "A#": "major", "C": "minor",
                       "D": "dim"},
                "E": {"E": "major", "F#": "minor", "G#": "minor", "A": "major", "B": "major", "C#": "minor",
                      "D#": "dim"},
                "F": {"F": "major", "G": "minor", "A": "minor", "A#": "major", "C": "major", "D": "minor", "E": "dim"},
                "F#": {"F#": "major", "G#": "minor", "A#": "minor", "B": "major", "C#": "major", "D#": "minor",
                       "E#": "dim"},
                "G": {"G": "major", "A": "minor", "B": "minor", "C": "major", "D": "major", "E": "minor", "F#": "dim"},
                "G#": {"G#": "major", "A#": "minor", "C": "minor", "C#": "major", "D#": "major", "F": "minor",
                       "G": "dim"},
                "A": {"A": "major", "B": "minor", "C#": "minor", "D": "major", "E": "major", "F#": "minor",
                      "G#": "dim"},
                "A#": {"A#": "major", "C": "minor", "D": "minor", "D#": "major", "F": "major", "G": "minor",
                       "A": "dim"},
                "B": {"D": "major", "C#": "minor", "D#": "minor", "E": "major", "F#": "major", "G#": "minor",
                      "A#": "dim"}}}
        self.notes_list = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    """Converter from midi number of note to letter and octave"""
    def converter_to_note(self, number):
        octave = (number // 12) - 1
        index = (number % 12)
        note = self.notes_list[index]
        return note, octave

    """Converter from letter and octave to midi number of note"""
    def converter_to_number(self, note, octave):
        number = self.notes_list.index(note) + (octave + 1) * 12
        return number

    """Converter from bemol note to analogical diez note"""
    def bemol_to_diez(self, list):
        for i in range(len(list)):
            if list[i] == "Bb":
                list[i] = "A#"
            elif list[i] == "Db":
                list[i] = "C#"
            elif list[i] == "Eb":
                list[i] = "D#"
            elif list[i] == "Gb":
                list[i] = "F#"
            elif list[i] == "Ab":
                list[i] = "G#"
            elif list[i] == "Cb":
                list[i] = "B"
            elif list[i] == "Fb":
                list[i] = "E"
            elif list[i] == "Bbb":
                list[i] = "A"
            elif list[i] == "Dbb":
                list[i] = "C"
            elif list[i] == "Abb":
                list[i] = "G"
        return list

    """Converter from diez note to analogical bemol note"""
    def diez_to_bemol(self, note):
        if note == "A#":
            return "Bb"
        elif note == "C#":
            return "Db"
        elif note == "D#":
            return "Eb"
        elif note == "F#":
            return "Gb"
        elif note == "G#":
            return "Ab"
        elif note == "A":  # TODO CHECK
            return "Bbb"
        elif note == "C":  # TODO CHECK
            return "Dbb"
        return note

    """Parse midi file and generate triad"""
    def parse(self, midi_file, key):
        melody = []

        """Parser of midi file to get all possible information about each note of melody"""
        for i, track in enumerate(midi_file.tracks):
            start_time = 0
            for msg in track:
                splited = str(msg).split(" ")
                if splited[0] == "note_on":
                    time = splited[4].split("=")[1]
                    if int(time) != 0:
                        melody.append({"type": "rest",
                                       "start_time": start_time,
                                       "finish_time": start_time + int(time)
                                       })
                        start_time += int(time)
                    note = self.converter_to_note(int(splited[2].split("=")[1]))
                    melody.append({"type": "note",
                                   "channel": splited[1].split("=")[1],
                                   "note_number": splited[2].split("=")[1],
                                   "note_name": note[0],
                                   "note_octave": note[1],
                                   "velocity": splited[3].split("=")[1],
                                   "start_time": start_time,
                                   "finish_time": 0})
                elif splited[0] == "note_off":
                    time = splited[4].split("=")[1]
                    melody[-1]["finish_time"] = melody[-1]["start_time"] + int(time)
                    start_time += int(time)

        """Generates triad for each note"""
        for element in melody:
            if element["type"] == "note":
                list_of_elements = self.minor_major_tables[key.mode][key.tonic.name]
                note = element["note_name"]
                if list_of_elements[note] == "minor":
                    triad_notes = chords.minor_triad(self.diez_to_bemol(note))
                elif list_of_elements[note] == "major":
                    triad_notes = chords.major_triad(self.diez_to_bemol(note))
                else:
                    triad_notes = chords.diminished_triad(self.diez_to_bemol(note))
                element["triad_notes"] = self.bemol_to_diez(triad_notes)
                element["triad_numbers"] = [
                    self.converter_to_number(chord, int(element["note_octave"]) - 1) % 12 for chord in
                    self.bemol_to_diez(triad_notes)]
        return melody


"""Class for evolutionary algorithm"""
class EvolutionaryAlgorithm:
    """
    population_size - number of individuals in population
    generation - number of iterations for algorithm to run
    children - number of children at each generation
    """
    def __init__(self, population_size, generation, children):
        self.population_size = population_size
        self.generation = generation
        self.children = children
        self.music_library = MusicLibrary()

    """Initializing the population"""
    def initial_population_generation(self, length):
        array = []
        chromosomes = self.population_size
        for i in range(chromosomes):
            list = []
            for j in range(length):
                trisound = []
                for k in range(3):
                    x = random.randint(0, 127) % 12
                    while x in trisound:
                        x = random.randint(0, 127) % 12
                    trisound.append(x)
                list.append(trisound)
            array.append(list)
        return array

    """Mutation of individual by swap"""
    def swap_mutation(self, chromosome):
        x = random.randint(0, len(chromosome) - 1)
        y = random.randint(0, len(chromosome) - 1)

        chromosome[x], chromosome[y] = chromosome[y], chromosome[x]

        a = random.randint(0, len(chromosome) - 1)
        b = random.randint(0, len(chromosome) - 1)
        c = random.randint(0, len(chromosome[0]) - 1)
        d = random.randint(0, len(chromosome[0]) - 1)

        chromosome[a][c], chromosome[b][d] = chromosome[b][d], chromosome[a][c]
        return chromosome

    """Crossover by one-point"""
    def crossover(self, chromosome_1, chromosome_2):
        x = random.randint(1, len(chromosome_1) - 1)
        x /= len(chromosome_1)
        tail = chromosome_1[int(len(chromosome_1) * x):]
        head = chromosome_2[:int(len(chromosome_2) * x)]
        head.extend(tail)
        return head

    """Metric for number of same notes in triad and trisound"""
    def match_distance(self, trisound, triad):
        distance = 0
        for note in trisound:
            if note in triad:
                distance += 1
        return distance / 3

    """Metric for Euclidian distance for different notes in triad and trisound"""
    def vector_distance(self, trisound, triad):
        repeated = []
        for note in trisound:
            if note in triad:
                repeated.append(note)

        triad_copy = triad.copy()
        trisound_copy = trisound.copy()
        if len(repeated) != 0:
            for note in repeated:
                while note in triad_copy:
                    triad_copy.remove(note)
                while note in trisound_copy:
                    trisound_copy.remove(note)

        unique_combinations = []
        permutations = itertools.permutations(trisound_copy, len(triad_copy))

        for combination in permutations:
            zipped = zip(combination, triad_copy)
            unique_combinations.append(list(zipped))

        min = -1
        for triplet in unique_combinations:
            a = []
            b = []
            for pair in triplet:
                a.append(pair[0])
                b.append(pair[1])
            d = self.euclidian_distance(a, b)
            if min == -1 or min > d:
                min = d

        return min / (12 * sqrt(3))

    """Euclidian distance between two vectors"""
    def euclidian_distance(self, a, b):
        return np.linalg.norm(np.asarray(a) - np.asarray(b))

    """Metric for same chords in one trisound"""
    def same_chords_distance(self, trisound):
        return (len(list(trisound)) - len(set(trisound))) / 3

    """Metric for distance between two nearest notes in trisound"""
    def length_distance(self, trisound):
        t = sorted(trisound)
        distance_1 = abs(t[1] - t[0]) if abs(t[1] - t[0]) % 12 < 12 - abs(t[1] - t[0]) % 12 else 12 - abs(t[1] - t[0]) % 12
        distance_2 = abs(t[2] - t[0]) if abs(t[2] - t[0]) % 12 < 12 - abs(t[2] - t[0]) % 12 else 12 - abs(t[2] - t[0]) % 12
        distance_3 = abs(t[2] - t[1]) if abs(t[2] - t[1]) % 12 < 12 - abs(t[2] - t[1]) % 12 else 12 - abs(t[2] - t[1]) % 12
        distances_list = [distance_1, distance_2, distance_3]
        distances_list = sorted(distances_list)
        if 3 <= distances_list[0] <= 4 and 3 <= distances_list[1] <= 4:
            return 0.5
        return -0.5

    """Metric for check of initial melody note in trisound"""
    def initial_value_distance(self, trisound, triad):
        if triad[0] in trisound:
            return 1
        return 0

    """Fitness function for one trisound"""
    def trisound_fitness(self, trisound, triad):
        w1 = 20
        w2 = -5
        w3 = -15
        w4 = 10
        w5 = 10
        d1 = self.match_distance(trisound, triad)
        d2 = self.vector_distance(trisound, triad)
        d3 = self.same_chords_distance(trisound)
        d4 = self.length_distance(trisound)
        d5 = self.initial_value_distance(trisound, triad)

        score = (w1 * d1 + w2 * d2 + w3 * d3 + w4 * d4 + w5 * d5)
        return score

    """Fitness function for one individual"""
    def individual_fitness(self, person, adam):
        score = 0
        for trisound, triad in zip(person, adam):
            score += self.trisound_fitness(trisound, triad)

        return score

    """Cut off individuals with smallest fitness score"""
    def replace_population(self, population, new, fitness, best):
        size = len(population)
        population.extend(new)
        population.sort(key=lambda x: fitness(x, best))

        return population[-size:], population[-1]

    """Run of crossover, mutation for all the generations"""
    def evolution(self, count, ideal):
        population = self.initial_population_generation(count)

        for i in range(self.generation):
            mothers = population[-2 * CHILDREN::2]
            fathers = population[-2 * CHILDREN + 1::2]
            children = []

            for mother, father in zip(mothers, fathers):
                child = self.swap_mutation(self.crossover(mother, father))
                children.append(child)

            population = self.replace_population(population, children,
                                                 self.individual_fitness, ideal)
            population, best = population[0], population[1]
        return best


mid = MidiFile(INPUT_FILE)

score: music21.stream.Score = music21.converter.parse(INPUT_FILE)
key = score.analyze('key')

output = MidiFile()
for i, track in enumerate(mid.tracks):
    output.tracks.append(track)
library = MusicLibrary()

melody = library.parse(mid, key)
time = 0
count = 0
ideal_sequence = []
start_time_of_ideal_sequence = []

"""Forming ideal sequence of triads and durations"""
for element in melody:
    if element["type"] == "note":
        if int(element["start_time"]) == time:
            ideal_sequence.append((element['triad_numbers']))
            start_time_of_ideal_sequence.append(element["start_time"])
            time += HALF_BAR
            count += 1
        elif int(element["start_time"]) > time:
            time = element["finish_time"]
    else:
        time += QUARTER_BAR

"""Run the Evolutionary Algorithm"""
EA = EvolutionaryAlgorithm(POPULATION_SIZE, GENERATION, CHILDREN)
result = EA.evolution(count, ideal_sequence)
velocity = [int(element["velocity"]) for element in melody if element['type'] == "note"]
velocity = int(sum(velocity) / len(velocity))
pause = []
end = []

"""Output in midi file"""
for i in range(3):
    moment_of_time = 0
    new_track = MidiTrack()
    output.tracks.append(new_track)
    new_track.append(MetaMessage('track_name', name='Elec. Piano (Classic)'+str(i), time=0))
    new_track.append(Message('program_change', program=0, time=0))
    current_time = 0
    flag = 0
    for j in range(count):
        if i == 0:
            duration = random.randint(0, 1)
            if flag == 0:
                new_track.append(
                    Message('note_on', note=result[j][i] + OCTAVE * 12, velocity=velocity, time=duration * QUARTER_BAR))
                pause.append(duration * QUARTER_BAR)
                current_time += duration*QUARTER_BAR
            else:
                new_track.append(
                    Message('note_on', note=result[j][i] + OCTAVE * 12, velocity=velocity,
                            time=start_time_of_ideal_sequence[t]-current_time))
                pause.append(start_time_of_ideal_sequence[t]-current_time)
                current_time+=start_time_of_ideal_sequence[t]-current_time
                flag = 0
            new_track.append(Message('note_off', note=result[j][i] + OCTAVE * 12, velocity=0,
                                     time=duration * QUARTER_BAR + (1 - duration) * HALF_BAR))
            end.append(duration * QUARTER_BAR + (1 - duration) * HALF_BAR)
            current_time += duration * QUARTER_BAR + (1 - duration) * HALF_BAR
            t = 0
            for k in range(len(start_time_of_ideal_sequence)):
                if current_time < start_time_of_ideal_sequence[k]:
                    t = k
                    break
            if start_time_of_ideal_sequence[t] - start_time_of_ideal_sequence[t - 1] > BAR:
                flag = 1

        else:
            new_track.append(Message('note_on', note=result[j][i] + OCTAVE * 12, velocity=velocity,
                                     time=pause[j]))
            new_track.append(Message('note_off', note=result[j][i] + OCTAVE * 12, velocity=0,
                                     time=end[j]))
    new_track.append(MetaMessage('end_of_track', time=0))

output.save(OUTPUT_FILE)