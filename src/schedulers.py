import math

class Scheduler():
    def __init__(self,
                 values,
                 total_steps,
                 schedule_type="linear",
                 warmup_steps=0,
                ):

        self.values = list(map(float, values))
        self.total_steps = float(total_steps)
        self.schedule_type = str(schedule_type)
        self.warmup_steps = float(warmup_steps)

        match self.schedule_type:
            case "constant":
                num_values = 1

            case "linear" | "cosine":
                num_values = 2

            case "warmup_linear" | "warmup_cosine":
                num_values = 3

            case _:
                raise ValueError(f"Unsupported scheduler type: {self.schedule_type}")

        if len(self.values) != num_values:
            raise ValueError(f"'{self.schedule_type}' requires {num_values} values.")

        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("Scheduler values must be finite.")

        if not math.isfinite(self.total_steps) or self.total_steps <= 0:
            raise ValueError("total_steps must be positive and finite.")

        if not math.isfinite(self.warmup_steps) or not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be non-negative and smaller than total_steps.")

        if not self.schedule_type.startswith("warmup_") and self.warmup_steps != 0:
            raise ValueError("warmup_steps requires a scheduler type with warmup.")

        self._step = 0
        self.actual_value = self._compute_value(self._step)

    def get_value(self):
        return self.actual_value

    def step(self):
        self._step += 1
        self.actual_value = self._compute_value(self._step)

        return self.actual_value

    def _compute_value(self, step):
        if self.schedule_type == "constant":
            return self.values[0]

        if step >= self.total_steps:
            return self.values[-1]

        start, final = self.values[0], self.values[-1]
        if self.schedule_type.startswith("warmup_"):
            start = self.values[1]
            if self.warmup_steps > 0 and step <= self.warmup_steps:
                progress = max(0.0, step / self.warmup_steps)
                return self.values[0] + progress * (start - self.values[0])

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = max(0.0, min(1.0, progress))

        match self.schedule_type:
            case "cosine" | "warmup_cosine":
                progress = 0.5 * (1.0 - math.cos(math.pi * progress))

        return start + progress * (final - start)

    def state_dict(self):
        return {
            "values": self.values.copy(),
            "total_steps": self.total_steps,
            "schedule_type": self.schedule_type,
            "warmup_steps": self.warmup_steps,
            "_step": self._step,
        }

    def load_state_dict(self, state_dict):
        for name in ["values", "total_steps", "schedule_type", "warmup_steps"]:
            if state_dict[name] != getattr(self, name):
                raise ValueError(f"Scheduler configuration differs for '{name}'.")

        step = int(state_dict["_step"])
        if step < 0:
            raise ValueError("Scheduler step cannot be negative.")

        self._step = step
        self.actual_value = self._compute_value(self._step)


class OptimizerScheduler(Scheduler):
    def __init__(self,
                 optimizer,
                 parameter,
                 values,
                 total_steps,
                 schedule_type="linear",
                 warmup_steps=0,
                 param_group_filter=None,
                ):

        self.optimizer = optimizer
        self.parameter = str(parameter)
        self.param_group_filter = param_group_filter

        for group in self.optimizer.param_groups:
            if self.parameter not in group:
                raise ValueError(f"Optimizer parameter group has no '{self.parameter}' field.")

        super().__init__(values, total_steps, schedule_type, warmup_steps)

        for group in self.optimizer.param_groups:
            group.setdefault(f"initial_{self.parameter}", group[self.parameter])

        self._apply()

    def group_uses_weight_decay(self, group):
        if group.get("is_bias", False) and not group.get("decay_bias", False):
            return False

        if group.get("is_norm", False) and not group.get("decay_norm", False):
            return False

        return True

    def _apply(self):
        for group in self.optimizer.param_groups:
            if self.param_group_filter is not None and not self.param_group_filter(group):
                continue

            if self.parameter == "weight_decay" and not self.group_uses_weight_decay(group):
                group[self.parameter] = 0.0
            elif group.get(f"fix_{self.parameter}", False):
                group[self.parameter] = group[f"initial_{self.parameter}"]
            else:
                scale = group.get(f"{self.parameter}_scale", 1.0)
                group[self.parameter] = self.actual_value * scale

    def step(self):
        value = super().step()
        self._apply()

        return value

    def state_dict(self):
        state = super().state_dict()
        state["parameter"] = self.parameter

        return state

    def load_state_dict(self, state_dict):
        if state_dict["parameter"] != self.parameter:
            raise ValueError("Optimizer scheduler parameter differs from the saved state.")

        super().load_state_dict(state_dict)
        self._apply()
