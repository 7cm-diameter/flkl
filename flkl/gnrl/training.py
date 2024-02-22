from amas.agent import Agent
from utex.clap import Config

from flkl.share import Flkl


async def flickr_discrimination(agent: Agent, ino: Flkl, expvars: dict):
    from amas.agent import NotWorkingError
    from numpy import arange
    from numpy.random import uniform
    from utex.agent import AgentAddress
    from utex.audio import Speaker, WhiteNoise
    from utex.scheduler import (SessionMarker, TrialIterator,
                                blockwise_shufflen, mix, repeat)

    from flkl.share import as_millis, flush_message_for

    reward_pin = expvars.get("reward-pin", 4)
    response_pin = expvars.get("response-pin", [6])
    audio_pin = expvars.get("audio-pin", 2)
    visual_pin = expvars.get("visual-pin", 3)

    noise = WhiteNoise()
    speaker = Speaker(expvars.get("speaker-id", 1))

    reward_duration = expvars.get("reward-duration", 0.02)
    flickr_duration = expvars.get("flickr-duration", 2.)

    reward_duration_millis = as_millis(reward_duration)
    flickr_duration_millis = as_millis(flickr_duration)

    reward_probability_for_audio = expvars.get("reward-probability-for-audio", .1)

    visual_target_frequency = expvars.get("traget-frequency", 6)
    visual_flickr_arange = tuple(expvars.get("visual-flickr-range", [-2., 2., .5]))
    visual_distracter_frequency = list(filter(lambda x: x != visual_target_frequency, visual_target_frequency + arange(*visual_flickr_arange)))
    visual_traget_flickrs = repeat(visual_target_frequency, len(visual_distracter_frequency))
    visual_flickrs = mix(
        visual_traget_flickrs,
        visual_distracter_frequency,
        expvars.get("target-ratio", 1),
        expvars.get("distracter-ratio", 1)
    )

    audio_flickr_arange = tuple(expvars.get("audio-flickr-range", [-1., 1., .5]))
    audio_distracter_frequency = list(filter(lambda x: x != visual_target_frequency, visual_target_frequency + arange(*audio_flickr_arange)))

    modality_block = mix(
        repeat([1], len(visual_flickrs)),
        repeat([0], len(audio_flickr_arange)),
        expvars.get("visual-ratio", 1),
        expvars.get("audio-ratio", 1)
    )
    flickr_block = mix(
        visual_flickrs,
        audio_distracter_frequency,
        expvars.get("visual-ratio", 1),
        expvars.get("audio-ratio", 1)
    )

    number_of_trial = expvars.get("number-of-trial", 500)
    blocksize = len(modality_block)
    flickr_per_trial, modality_per_trial = blockwise_shufflen(
        blocksize,
        repeat(flickr_block, number_of_trial // blocksize),
        repeat(modality_block, number_of_trial // blocksize),
    )

    mean_interval = expvars.get("ITI", 5.)
    range_interval = expvars.get("ITI-range", 0.)
    itis = uniform(
        mean_interval - range_interval,
        mean_interval + range_interval,
        number_of_trial
    )

    trials = TrialIterator(
        flickr_per_trial,
        modality_per_trial,
        itis
    )

    try:
        while agent.working():
            speaker.play(noise, False, True)
            for i, freq, is_visual, iti in trials:
                print(f"Trial {i}")
                if is_visual:
                    rewarded = freq == visual_target_frequency
                    ino.flick_for(visual_pin, freq, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration)
                    if rewarded:
                        ino.high_for(reward_pin, reward_duration_millis)
                        await flush_message_for(agent, reward_duration)
                    else:
                        await flush_message_for(agent, reward_duration)
                else:
                    rewarded = uniform() <= reward_probability_for_audio
                    ino.flick_for(audio_pin, freq, flickr_duration_millis)
                    await flush_message_for(agent, flickr_duration)
                    if rewarded:
                        ino.high_for(reward_pin, reward_duration_millis)
                        await flush_message_for(agent, reward_duration)
                    else:
                        await flush_message_for(agent, reward_duration)
            speaker.stop()
            agent.send_to(AgentAddress.OBSERVER.value, SessionMarker.NEND)
            agent.finish()
    except NotWorkingError:
        pass
