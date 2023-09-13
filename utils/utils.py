from robopianist import suite
from robopianist import music
from mujoco_utils import composer_utils
from robopianist.suite.tasks import piano_with_shadow_hands 
import note_seq
import scipy
from robopianist import suite
from robopianist import music
from mujoco_utils import composer_utils
from robopianist.suite.tasks import piano_with_shadow_hands 
import note_seq
from robopianist.music import midi_file

def truncate_note_sequence(sequence, start_time, end_time):
    """
    Modified from https://github.com/magenta/note-seq/blob/main/note_seq/sequences_lib.py: 
    note_seq.trim_note_sequence() 
    Instead of skipping notes that started earlier than start_time, truncate_note_sequence truncates them.
    """

    from note_seq.protobuf import music_pb2
    subsequence = music_pb2.NoteSequence()
    #subsequence.CopyFrom(sequence)

    #del subsequence.notes[:] 

    for note in sequence.notes:
        if note.end_time < start_time or note.start_time >= end_time:
            continue
        new_note = subsequence.notes.add()
        new_note.CopyFrom(note)
        new_note.start_time = max(start_time, note.start_time) - start_time
        new_note.end_time = min(note.end_time, end_time) - start_time

    #subsequence.total_time = min(sequence.total_time, end_time-start_time)
    subsequence.total_time = end_time-start_time
    
    return subsequence

def get_overlapping_chunks(midi, length, overlap_left, overlap_right):
    
    chunks = []

    i = 0
    
    while i < midi.duration:
        
        start = max(0, i-overlap_left)
        end = min (i+length+overlap_right, midi.duration)
        
        chunk = truncate_note_sequence(midi.seq, start , end)

        #print(f"{i}: [{start} - {end}] #notes: {len(chunk.notes)} ({chunk.total_time})")
        #note_seq.play_sequence(chunk)
        
        midi_segment = midi_file.MidiFile.from_note_seq(chunk) 
        #print(f"duration: {midi_segment.duration} has fingering?: {midi_segment.has_fingering()}")
        
        chunks.append(midi_segment)
        i+=length

    return chunks

def get_overlapping_chunks_with_auto_overlap(midi, length):
    import numpy as np
    
    
    num_chunks = int(np.ceil(midi.duration/length))
    #total_shift = num_chunks*length - midi.duration
    start_shift = ((num_chunks*length - midi.duration)/ (num_chunks-1)) #no overlap left for the first segment, no overlap right for the last segment
    #overlap_len = total_overlap#/(num_overlaps*2)

    
    #print("input duration: ", midi.duration)
    #print("#chunks", num_chunks)
    #print(total_shift)
    #print("start_shift",   start_shift)
    chunks = []

  
    for i in range(num_chunks):
        
        start = max(0, i*(length - start_shift))
        end = min (start+length, midi.duration)
        
        chunk = truncate_note_sequence(midi.seq, start , end)

        #print(f"\t[{start} -> {end}] #notes: {len(chunk.notes)} ({chunk.total_time})")
        #note_seq.play_sequence(chunk)
        
        midi_segment = midi_file.MidiFile.from_note_seq(chunk) 
        #print(f"\t\tduration: {midi_segment.duration} has fingering?: {midi_segment.has_fingering()}")
        
        chunks.append(midi_segment)
        

    return chunks



env_names_train_set_32_1 = ['RoboPianist-repertoire-150-EtudeOp10No3-v0',
 'RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo23InBMajor-v0',
 'RoboPianist-repertoire-150-EtudeOp25No11-v0',
 'RoboPianist-repertoire-150-BalladeNo1-v0',
 'RoboPianist-repertoire-150-EtudeOp10No12-v0',
 'RoboPianist-repertoire-150-PreludeOp28No7-v0',
 'RoboPianist-repertoire-150-PianoSonataK332InFMajor3RdMov-v0',
 'RoboPianist-repertoire-150-ClairDeLune-v0',
 'RoboPianist-repertoire-150-PianoSonataK576InDMinor1StMov-v0',
 'RoboPianist-repertoire-150-MusicalMomentOp16No4-v0',
 'RoboPianist-repertoire-150-TwoPartInventionInDMajor-v0',
 'RoboPianist-repertoire-150-PianoSonataK282InEbMajorMinuet1-v0',
 'RoboPianist-repertoire-150-FantaisieImpromptu-v0',
 'RoboPianist-repertoire-150-PianoSonataNo142NdMov-v0',
 'RoboPianist-repertoire-150-VenetianischesGondelliedOp30No6-v0',
 'RoboPianist-repertoire-150-RomanianDanceNo1-v0',
 'RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo7InEbMajor-v0',
 'RoboPianist-repertoire-150-PianoSonataK281InBbMajor1StMov-v0',
 'RoboPianist-repertoire-150-CarnivalOp37ANo2-v0',
 'RoboPianist-repertoire-150-ImpromptuOp90No4-v0',
 'RoboPianist-repertoire-150-PianoSonataNo141StMov-v0',
 'RoboPianist-repertoire-150-PianoSonata1StMov-v0',
 'RoboPianist-repertoire-150-WaltzOp64No2-v0',
 'RoboPianist-repertoire-150-FrohlicherLandmannOp68No10-v0',
 'RoboPianist-repertoire-150-LaChasseOp19No3-v0',
 'RoboPianist-repertoire-150-PreludeOp23No5-v0',
 'RoboPianist-repertoire-150-EnglishSuiteNo2Prelude-v0',
 'RoboPianist-repertoire-150-FantasieStuckeOp12No7-v0',
 'RoboPianist-repertoire-150-TwoPartInventionInCMajor-v0',
 'RoboPianist-repertoire-150-MazurkaOp7No1-v0',
 'RoboPianist-repertoire-150-ScherzoNo2-v0',
 'RoboPianist-repertoire-150-PartitaNo42-v0']

env_names_train_set_64_1 = ['RoboPianist-repertoire-150-EtudeOp10No3-v0',
 'RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo23InBMajor-v0',
 'RoboPianist-repertoire-150-EtudeOp25No11-v0',
 'RoboPianist-repertoire-150-BalladeNo1-v0',
 'RoboPianist-repertoire-150-EtudeOp10No12-v0',
 'RoboPianist-repertoire-150-PreludeOp28No7-v0',
 'RoboPianist-repertoire-150-PianoSonataK332InFMajor3RdMov-v0',
 'RoboPianist-repertoire-150-ClairDeLune-v0',
 'RoboPianist-repertoire-150-PianoSonataK576InDMinor1StMov-v0',
 'RoboPianist-repertoire-150-MusicalMomentOp16No4-v0',
 'RoboPianist-repertoire-150-TwoPartInventionInDMajor-v0',
 'RoboPianist-repertoire-150-PianoSonataK282InEbMajorMinuet1-v0',
 'RoboPianist-repertoire-150-FantaisieImpromptu-v0',
 'RoboPianist-repertoire-150-PianoSonataNo142NdMov-v0',
 'RoboPianist-repertoire-150-VenetianischesGondelliedOp30No6-v0',
 'RoboPianist-repertoire-150-RomanianDanceNo1-v0',
 'RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo7InEbMajor-v0',
 'RoboPianist-repertoire-150-PianoSonataK281InBbMajor1StMov-v0',
 'RoboPianist-repertoire-150-CarnivalOp37ANo2-v0',
 'RoboPianist-repertoire-150-ImpromptuOp90No4-v0',
 'RoboPianist-repertoire-150-PianoSonataNo141StMov-v0',
 'RoboPianist-repertoire-150-PianoSonata1StMov-v0',
 'RoboPianist-repertoire-150-WaltzOp64No2-v0',
 'RoboPianist-repertoire-150-FrohlicherLandmannOp68No10-v0',
 'RoboPianist-repertoire-150-LaChasseOp19No3-v0',
 'RoboPianist-repertoire-150-PreludeOp23No5-v0',
 'RoboPianist-repertoire-150-EnglishSuiteNo2Prelude-v0',
 'RoboPianist-repertoire-150-FantasieStuckeOp12No7-v0',
 'RoboPianist-repertoire-150-TwoPartInventionInCMajor-v0',
 'RoboPianist-repertoire-150-MazurkaOp7No1-v0',
 'RoboPianist-repertoire-150-ScherzoNo2-v0',
 'RoboPianist-repertoire-150-PartitaNo42-v0',
 'RoboPianist-repertoire-150-PianoSonataNo82NdMov-v0',
 'RoboPianist-repertoire-150-JeuxDeau-v0',
 'RoboPianist-repertoire-150-PianoSonataNo143RdMov-v0',
 'RoboPianist-repertoire-150-PavanePourUneInfanteDefunte-v0',
 'RoboPianist-repertoire-150-PianoSonataNo213RdMov-v0',
 'RoboPianist-repertoire-150-SuiteEspanolaOp45No1-v0',
 'RoboPianist-repertoire-150-PianoSonataK576InDMajor2NdMov-v0',
 'RoboPianist-repertoire-150-LyricPiecesOp43No1-v0',
 'RoboPianist-repertoire-150-PianoSonataK284InDMajor1StMov-v0',
 'RoboPianist-repertoire-150-WellTemperedClavierBookIiPreludeNo11InFMajor-v0',
 'RoboPianist-repertoire-150-PeerGyntOp46No2-v0',
 'RoboPianist-repertoire-150-PianoSonataK457InCMinor3RdMov-v0',
 'RoboPianist-repertoire-150-KreislerianaOp16No1-v0',
 'RoboPianist-repertoire-150-PianoSonataK283InGMajor1StMov-v0',
 'RoboPianist-repertoire-150-PianoSonataNo303RdMov-v0',
 'RoboPianist-repertoire-150-TwoPartInventionInCMinor-v0',
 'RoboPianist-repertoire-150-WellTemperedClavierBookIPreludeNo2InCMinor-v0',
 'RoboPianist-repertoire-150-EnglishSuiteNo3Prelude-v0',
 'RoboPianist-repertoire-150-SonataInAMajorK208-v0',
 'RoboPianist-repertoire-150-KreislerianaOp16No3-v0',
 'RoboPianist-repertoire-150-JeTeVeux-v0',
 'RoboPianist-repertoire-150-PianoSonataNo5-v0',
 'RoboPianist-repertoire-150-NocturneOp9No2-v0',
 'RoboPianist-repertoire-150-SongWithoutWordsOp19No1-v0',
 'RoboPianist-repertoire-150-SinfoniaNo12-v0',
 'RoboPianist-repertoire-150-ItalianConverto1StMov-v0',
 'RoboPianist-repertoire-150-Reverie-v0',
 'RoboPianist-repertoire-150-PianoSonataK284InDMajor3RdMov-v0',
 'RoboPianist-repertoire-150-SuiteBergamasquePrelude-v0',
 'RoboPianist-repertoire-150-PianoSonataK570InBbMajor1StMov-v0',
 'RoboPianist-repertoire-150-PianoSonataNo43RdMov-v0',
 'RoboPianist-repertoire-150-GrandeValseBrillanteOp18-v0']


def get_all_training_melodies():

    from robopianist import music

    pig_melodies = set(music.PIG_MIDIS) - set(music.ETUDE_MIDIS) 

    #Filter Songs with empty notes

    filter_out = set()
    for melody in pig_melodies:
        midi = music.load(melody)

        for note in midi.seq.notes:
            if (note.end_time-note.start_time) <= 0:
                filter_out.add(melody)
                break 

    pig_melodies = pig_melodies - filter_out
                
    
    return list(pig_melodies), list(filter_out)

def get_all_training_envs():

    environment_name = "RoboPianist-repertoire-150-{}-v0"

    names, _ = get_all_training_melodies()
    environment_names = [environment_name.format(name) for name in names]
    return environment_names