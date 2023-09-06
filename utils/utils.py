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
