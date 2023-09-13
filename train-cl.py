from pathlib import Path
from typing import Optional, Tuple
import tyro
from dataclasses import dataclass, asdict
import wandb
import time
import random
import numpy as np
from tqdm import tqdm
import orbax.checkpoint

import sac
import specs
import replay

from robopianist import suite
import dm_env_wrappers as wrappers
import robopianist.wrappers as robopianist_wrappers

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

from collections import defaultdict

@dataclass(frozen=True)
class Args:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    curriculum: str = "uniform" #uniform, inverse_reward
    alpha: float = 1 #curriculum exploration factor
    segment_core_length: float = 2.8
    overlap_left: float = 0.1
    overlap_right: float = 0.1
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "disabled"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    train_environment_names: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    test_environment_names: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    agent_config: sac.SACConfig = sac.SACConfig()
    randomize_hand_positions: bool = False


def prefix_dict(prefix: str, d: dict) -> dict:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def melody_name_from_env_name(env_name: str) -> str:
    return env_name.split('-')[-2]

    
def get_env(environment_name, args: Args, callback=None, record_dir: Optional[Path] = None):

    env = suite.load(
        environment_name=environment_name,
        callback=callback,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
            randomize_hand_positions=args.randomize_hand_positions
        ),
    )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
        env = wrappers.EpisodeStatisticsWrapper(
            environment=env, deque_size=args.record_every
        )
        env = robopianist_wrappers.MidiEvaluationWrapper(
            environment=env, deque_size=args.record_every
        )
    else:
        env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(
            env, num_frames=args.frame_stack, flatten=True
        )
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


class UniformCurriculum():

    def __init__(self, names=[]) -> None:
        self.names = names 

        self.cur = 0 
        self.called = 0 


        #initialize midis
        self.midis = []

        for name in self.names:
            midi = music.load(name)
            self.midis.append(midi)

    def get_midi(self):
        
        midi = self.midis[self.cur]
        self.cur = (self.cur+1)%len(self.midis)
        self.called += 1
        #print(f"Called get_midi. Cur: {self.cur} #call: {self.called}")
        return midi



from utils import utils


class SegmentCurriculum():

    def __init__(self, names, args, mode="uniform", segment_core_length=2.8, overlap_left=0.1, overlap_right=0.1) -> None:
        self.names = names 
        self.mode = mode 

        self.cur = 0 
        self.called = 0 
    


        #initialize midis
        self.original_midis = []
        self.segments = []

        for name in self.names:
            midi = music.load(name)
            self.original_midis.append(midi)
            #segments = get_overlapping_chunks(midi, length=segment_core_length, overlap_left=overlap_left, overlap_right=overlap_right)
            segments = utils.get_overlapping_chunks_with_auto_overlap(midi, length=segment_core_length)
            self.segments.extend(segments)

        # Segment Stat

        total_original_len = sum([midi.duration for midi in self.original_midis])
        total_segments_len = sum([midi.duration for midi in self.segments])
        mean_segments_len = np.mean([midi.duration for midi in self.segments])

        print(f"Segment Statistics:")
        print(f"From {len(self.original_midis)}({total_original_len} s) full midis created {len(self.segments)} segments(Mean: {mean_segments_len}s. Total {total_segments_len}s)")


        # Curriculum Data Structures

        eps = np.finfo(float).eps
        self.frequency     = np.zeros(len(self.segments))
        self.last_reward   = np.ones(len(self.segments)) * eps




        self.args =  args

    def update_curriculum(self, env_stat, step):
        
        rew = env_stat["return"]
        self.last_reward[self.cur] = rew

        #print(f"in update: \ncurrent_segment: {self.cur}. reward: {rew}")

        #for inverse-reward-proportional 
        if self.mode == "inverse_reward":
            p = scipy.special.softmax(self.args.alpha * 1 / self.last_reward)
            #print("softax:", p)
            self.cur = np.random.choice(np.arange(len(self.last_reward)), p=p)

            #print(f"new_segment: {self.cur} P = {p[self.cur]}")

        elif self.mode == "uniform":
            self.cur = np.random.randint(0, high=len(self.segments), dtype=int)
            #print(f"new_segment: {self.cur}")

        elif self.mode == "incremental_uniform":

            min_window_len = 80
            inc_start_tstep =  500000
            inc_end_tstep   = 3000000


            if step <= inc_start_tstep: window_len = min_window_len
            elif step >= inc_end_tstep: window_len = len(self.segments)
            else:
                
                slope = (len(self.segments) - min_window_len) / (inc_end_tstep-inc_start_tstep)
                increment = int(slope * (step-inc_start_tstep) )
                window_len = min_window_len + increment



            self.cur = np.random.randint(0, high=window_len, dtype=int) 

            #print("cur segment#index: ", self.cur, "window len", window_len) 
        else:
            raise NotImplementedError

        
        self.frequency[self.cur] += 1

    def get_midi(self):

        self.called += 1
        #self.cur = np.random.randint(0, high=len(self.segments), dtype=int)

        #print(f"Called get_midi. Cur: {self.cur} #call: {self.called}")

        return self.segments[self.cur]
        
def main(args: Args) -> None:
    if args.name:
        run_name = f"{args.name}-{args.seed}-{time.time()}"
    else:
        run_name = f"CL-SAC-{args.curriculum}-{args.seed}-{time.time()}"

    # Create experiment directory.
    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)

    # Seed RNGs.
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )


    env_names_etude_12 = ['RoboPianist-etude-12-FrenchSuiteNo1Allemande-v0', 'RoboPianist-etude-12-FrenchSuiteNo5Sarabande-v0', 
    'RoboPianist-etude-12-PianoSonataD8451StMov-v0', 'RoboPianist-etude-12-PartitaNo26-v0', 
    'RoboPianist-etude-12-WaltzOp64No1-v0', 'RoboPianist-etude-12-BagatelleOp3No4-v0',
    'RoboPianist-etude-12-KreislerianaOp16No8-v0', 'RoboPianist-etude-12-FrenchSuiteNo5Gavotte-v0', 
    'RoboPianist-etude-12-PianoSonataNo232NdMov-v0', 'RoboPianist-etude-12-GolliwoggsCakewalk-v0', 
    'RoboPianist-etude-12-PianoSonataNo21StMov-v0', 'RoboPianist-etude-12-PianoSonataK279InCMajor1StMov-v0']

    env_names_train_set_1 = ['RoboPianist-repertoire-150-PreludeOp28No19-v0', 'RoboPianist-repertoire-150-NorwegianDanceOp35No3-v0', 
            'RoboPianist-repertoire-150-PianoSonataNo41StMov-v0', 'RoboPianist-repertoire-150-NocturneOp9No2-v0',
            'RoboPianist-repertoire-150-BalladeNo2-v0', 'RoboPianist-repertoire-150-BalladeNo1-v0', 
            'RoboPianist-repertoire-150-PianoSonataNo5-v0', 'RoboPianist-repertoire-150-TwoPartInventionInCMinor-v0', 
            'RoboPianist-repertoire-150-LaChasseOp19No3-v0', 'RoboPianist-repertoire-150-PianoSonataK282InEbMajorMinuet1-v0', 
            'RoboPianist-repertoire-150-KreislerianaOp16No1-v0', 'RoboPianist-repertoire-150-LaFilleAuxCheveuxDeLin-v0']

    env_names_train_set_2 = ['RoboPianist-repertoire-150-MazurkaOp7No1-v0', 'RoboPianist-repertoire-150-SuiteBergamasquePasspied-v0', 
            'RoboPianist-repertoire-150-RomanianDanceNo1-v0', 'RoboPianist-repertoire-150-PianoSonataNo303RdMov-v0', 
            'RoboPianist-repertoire-150-Sonatine1StMov-v0', 'RoboPianist-repertoire-150-LaFilleAuxCheveuxDeLin-v0', 
            'RoboPianist-repertoire-150-PianoSonataNo241StMov-v0', 'RoboPianist-repertoire-150-LyricPiecesOp62No2-v0', 
            'RoboPianist-repertoire-150-JeuxDeau-v0', 'RoboPianist-repertoire-150-TwoPartInventionInCMinor-v0', 
            'RoboPianist-repertoire-150-PianoSonataNo43RdMov-v0', 'RoboPianist-repertoire-150-ForElise-v0']

    
    def get_env_names(names):
        if names == "etude_12":
            return env_names_etude_12
        elif names == "train_set_1": return env_names_train_set_1
        elif names == "train_set_2": return env_names_train_set_2
        elif names == 'train_set_all': return utils.get_all_training_envs()
        elif names == "train_set_64_1": return utils.env_names_train_set_64_1
        elif names == "train_set_32_1": return utils.env_names_train_set_32_1
        else:
            return [s.strip() for s in names.split(',')]
        
    
    train_env_names = get_env_names(args.train_environment_names)
    test_env_names = get_env_names(args.test_environment_names)
    
    train_melodies = [melody_name_from_env_name(s) for s in train_env_names]

    #etude_melodies = ["FrenchSuiteNo1Allemande", "FrenchSuiteNo5Sarabande", "PianoSonataD8451StMov", "PartitaNo26", 
    #              "WaltzOp64No1", "BagatelleOp3No4", "KreislerianaOp16No8", "FrenchSuiteNo5Gavotte", 
    #              "PianoSonataNo232NdMov", "GolliwoggsCakewalk", "PianoSonataNo21StMov", "PianoSonataK279InCMajor1StMov"]

    curriclulum = SegmentCurriculum(train_melodies,
                                        args=args,
                                        mode=args.curriculum,
                                        segment_core_length=args.segment_core_length, 
                                        overlap_left=args.overlap_left, 
                                        overlap_right=args.overlap_left
                                    )

    env = get_env(train_env_names[0], args, callback=curriclulum)

    eval_envs = {}
    #print("Debug: for now only 3 envs are tested")
    for env_name in test_env_names:
        eval_env = get_env(env_name, args, record_dir=experiment_dir / "eval")
        eval_envs[melody_name_from_env_name(env_name)]= eval_env

    spec = specs.EnvironmentSpec.make(env)

    agent = sac.SAC.initialize(
        spec=spec,
        config=args.agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1)
    checkpointer = orbax.checkpoint.CheckpointManager(
        experiment_dir / "checkpoints",
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        options=options,
    )

    timestep = env.reset()
    replay_buffer.insert(timestep, None)
 
    start_time = time.time()
    for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        # Act.
        if i < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        # Observe.
        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        # Reset episode.
        if timestep.last():

            env_stat = env.get_statistics()
            wandb.log(prefix_dict("train", env_stat), step=i)

            #update curriculum
            curriclulum.update_curriculum(env_stat=env_stat, step=i)

            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        # Train.
        if i >= args.warmstart_steps:
            if replay_buffer.is_ready():
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
                if i % args.log_interval == 0:
                    wandb.log(prefix_dict("train", metrics), step=i)

        # Eval.

        all_log_dicts = {}
        all_music_dicts = {}
        all_dicts = {}

        if i % args.eval_interval == 0:

            for env_name, eval_env in eval_envs.items():
                for _ in range(args.eval_episodes):
                    timestep = eval_env.reset()
                    while not timestep.last():
                        timestep = eval_env.step(agent.eval_actions(timestep.observation))

                # log_dict = prefix_dict("eval", eval_env.get_statistics())
                # music_dict = prefix_dict("eval", eval_env.get_musical_metrics())
                # wandb.log(log_dict | music_dict, step=i)
                video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
                wandb.log({f"video/{env_name}": video}, step=i)
                eval_env.latest_filename.unlink()

                all_log_dicts[env_name] = eval_env.get_statistics()
                all_music_dicts[env_name] = eval_env.get_musical_metrics()
                all_dicts[env_name] = all_log_dicts[env_name] | all_music_dicts[env_name]

            #log stat
            stat_dict = {}
            agg_stat_dict = defaultdict(list)

            for env_name, dict in all_dicts.items():
                for k, v in dict.items():

                    stat_dict[f"eval-{env_name}/{k}"] = v
                    agg_stat_dict[k].append(v)

            safe_mean = lambda l: np.mean(l)
            safe_std  = lambda l: np.std(l)

            for k, ls in agg_stat_dict.items():
                
                stat_dict[f"eval/{k}"] = safe_mean(ls)
                stat_dict[f"eval/{k}-std"] = safe_std(ls)
                


            wandb.log(stat_dict, step=i)
            checkpointer.save(i, agent)





        if i % args.log_interval == 0:
            wandb.log({"train/fps": int(i / (time.time() - start_time))}, step=i)

    checkpointer.save(i, agent)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
