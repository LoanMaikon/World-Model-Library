from collections import deque
import numpy as np
import torch

from stable_worldmodel.policy import WorldModelPolicy as BaseWorldModelPolicy

class WorldModelPolicy(BaseWorldModelPolicy):
    def set_env(self, env):
        super().set_env(env)
        self._frame_history = None

    def get_action(self, info_dict, **kwargs):
        actions = super().get_action(info_dict, **kwargs)
        return np.clip(actions, self.env.action_space.low, self.env.action_space.high)

    def _prepare_info(self, info_dict):
        info = super()._prepare_info(info_dict)
        context = self.cfg.history_len
        if context == 1:
            return info

        pixels = info["pixels"]
        count = len(pixels)
        block = self.cfg.action_block
        span = (context - 1) * block
        if getattr(self, "_frame_history", None) is None:
            self._frame_history = [deque(maxlen=span + 1) for _ in range(count)]
            self._action_history = [deque(maxlen=span) for _ in range(count)]
            self._last_ids = [None] * count
            self._last_steps = [None] * count

        def scalar(key, index):
            value = info.get(key)
            if value is None:
                return None
            value = value[index]
            if torch.is_tensor(value):
                return value.reshape(-1)[-1].item()
            return np.asarray(value).reshape(-1)[-1].item()

        incoming = info.get("action")
        histories, past_actions, lengths = [], [], []
        flush = np.zeros(count, dtype=bool)
        for index in range(count):
            frames = self._frame_history[index]
            actions = self._action_history[index]
            episode = scalar("id", index)
            step = scalar("step_idx", index)
            previous_id = self._last_ids[index]
            previous_step = self._last_steps[index]
            command = incoming[index, -1].reshape(-1) if incoming is not None else None
            reset = bool(scalar("_needs_flush", index))
            if frames:
                metadata_changed = (
                    episode is not None and previous_id is not None and episode != previous_id
                ) or (
                    step is not None and previous_step is not None and step < previous_step
                )
                reset |= metadata_changed and len(frames) > 1
                reset |= command is not None and not bool(torch.isfinite(command).all())
            if reset:
                frames.clear()
                actions.clear()
                flush[index] = True

            duplicate = (
                bool(frames)
                and step is not None
                and step == previous_step
                and episode == previous_id
            )
            if not duplicate:
                if frames:
                    if command is None:
                        raise ValueError("Observation history requires the previously executed action.")
                    actions.append(command.detach().clone())
                frames.append(pixels[index, -1].detach().clone())
            self._last_ids[index] = episode
            self._last_steps[index] = step

            length = min(context, 1 + len(actions) // block)
            selected = list(frames)[-(length - 1) * block - 1::block]

            histories.append(torch.stack([selected[0]] * (context - length) + selected))
            action_dim = incoming.shape[-1] if incoming is not None else self.env.single_action_space.shape[-1]
            previous = pixels.new_zeros((context - 1, block * action_dim))
            if length > 1:
                previous[-(length - 1):] = torch.stack(list(actions)[-(length - 1) * block:]).reshape(
                    length - 1, block * action_dim
                )
            past_actions.append(previous)
            lengths.append(length)

        info["pixels"] = torch.stack(histories)
        info["past_action"] = torch.stack(past_actions)
        info["history_length"] = torch.tensor(lengths, dtype=torch.long, device=pixels.device)
        if flush.any():
            info["_needs_flush"] = torch.as_tensor(flush)
        return info
