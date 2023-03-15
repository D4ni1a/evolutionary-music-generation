# Evolutionary algorithm for music generation

## Manual

Python version: _Python 3.9._

To run the program it is necessary to have several libraries installed on the computer.

The standard ones are: **math** , **random** , **itertools**.

The following libraries may need to be installed (if not installed already): **numpy** , **mido** ,
**music21** , **mingus**

**_pip install mingus_**

**_pip install mido_**

**_pip install numpy_**

**_pip install music_**

To generate a new sound accompaniment one just have to change the **INPUT_FILE** and
**OUTPUT_FILE** variables with the input and output midi files, and run the program itself. The
parameters that can be changed are the population size, the number of generations, the
number of children born in the new generation, and the output octave number.

## Key detection

I used the functions of the **music21** library for key detection, including the standard **'key'**
identification algorithm. **music21** allows to automatically run an analysis to get the key of a
piece or excerpt not based on the key signature but instead on the frequency with which some
notes are used as opposed to others.

As a result, I got the following keys for the 3 input files:

**input1.mid** - D minor

**input2.mid** - F major

**input3.mid** - E minor

## Description of algorithm

My program can be logically divided into 2 major blocks: parsing and processing of midi files
(represented by class **MusicLibrary** ) and the evolutionary algorithm for generating an
accompaniment (represented by class **EvolutionaryAlgorithm** ).

## Parsing and processing of midi files

After detecting a melody key, using the Minor/Major key table (represented as a
**minor_major_tables** dictionary), it is possible to determine Minor/Major/Diminished triads


(depending on a key and on a note itself) for each note of a melody. These triads, as well as
information about the note itself, octave, velocity, etc., are saved in a special dictionary.

In addition, the **MusicLibrary** class contains functions for converting a note from alphabetic
notation to midi number and vice versa and for converting notes that are not contained in the
**minor_major_tables** dictionary (for simplification, some notes are not listed there, e.g. _Bb_ ,
_Cb_ ,...).

## Evolutionary algorithm

As a single member of the population we considered a sequence of trisounds of fixed length
(the length is calculated from the analysis of the original melody), i.e. it is a matrix of size N x 3,
where N is the length of the sequence.

The main components are:

- Initial population generation;
- Fitness function;
- Mutation;
- Crossover;
- Generating a new population.

_Generating the initial population_

The number of individuals in the population is initially set with the variable **POPULATION_SIZE**.
Several matrices of size N x 3 (i.e. several individuals) are generated, each note of which is a
remainder of division by 12 of a random integer in the range from 0 to 127. In addition, we
check that there are no duplicate notes in the generated trisounds, and if there are, the
duplicates are regenerated.

_Fitness Function_

The basic idea is that for some notes in a melody, one can compose a Minor/Major/Diminished
triad and use it as the optimal value. In other words, take the triad as an ideal to which all triad
sequences should aspire. The fitness score of an individual consists of the sum of the fitness
scores of each trisound. In turn, the fitness score of a trisound consists of the sum of several
individual measurements multiplied by weights, depending on the importance of that value
(based on my subjective evaluation).

_Measurements_ :

**d1** - Calculates how many of the triad notes consist in a trisound. The more, the better;

**d2** - Measures the " _Euclidean distance_ " between all possible combinations of three "trisound
note - triad note" pairs and chooses the smallest of them. If a triad and a trisound have the
same notes, they are not included in the measurement. The smaller the distance, the better;

**d3** - Calculates how many identical notes are in a trisound. The smaller, the better;


**d4** - Since neighboring notes in a triad share 3 or 4 orders (depending on the note and triad), it
would be nice if the neighboring notes in the triad were also not too close or too far apart.
Counts whether there are any notes in the triad that are too close or too far apart. If there are
none, that's better;

**d5** - Checks whether the note of the original melody is in a triad in a different octave. If there is,
it's better.

Before multiplying by weights, all values are normalized, which means reduced to a number
between 0 and 1, by dividing by the largest possible (or just large enough) number for a given
metric.

The greatest positive weight has metric **d1** , and the greatest negative weight has metric **d**.

All calculations are made with values without octave reference. That is, all notes are in the
range of 0 to 11.

_Crossover_

I used the one-point as the crossover algorithm. For two parents, one point is randomly
determined to be exchanged. The two individuals exchange some sequence of trisound and the
result is one new sequence.

_Mutation_

I used swap as the mutation algorithm. Two trisounds in the sequence are randomly chosen
and swapped. After that, two other random trisounds are chosen, each of them has one of the
three notes chosen at random, and those notes are swapped.

_Forming a new population_

The number of generations is determined at the beginning with a variable. Then, after
crossover and mutation with some number of individuals, a set of new individuals is formed,
and the fitness function is used to cut off individuals with the smallest fitness score, so that in
the end there are as many individuals as there were in the population at the beginning.

After all iterations, the individual with the highest fitness score is extracted. The initial melody
is written to the output file. After that, 3 new tracks are added there, each of which is a
sequence of corresponding notes in trisounds. The duration of rest and notes is determined
depending on the rests in the initial melody, as well as there is added an element of
randomness (as if the mutation in duration) changing the duration of the note from a quarter
(in this case rest is added in the beginning) to half the duration of bar.


