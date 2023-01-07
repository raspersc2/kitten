"""
Run independently of bot
Run game with `OfflineAgent` to collect data
Then use this to load model, data and perform back propagation and save the model
"""
from os import listdir, makedirs, path
from pathlib import Path

import numpy as np
import torch
from typing import Dict, List

import yaml
from loguru import logger
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter

from bot.consts import SQUAD_ACTIONS, ConfigSettings
from bot.squad_agent.architecture.actor_critic import ActorCritic
from bot.squad_agent.utils import load_checkpoint


SPATIAL_SHAPE: tuple[int, int, int, int] = (1, 37, 120, 120)
ENTITY_SHAPE: tuple[int, int, int] = (1, 256, 406)
SCALAR_SHAPE: tuple[int, int] = (1, 8)


class PPOTrainer:
    def __init__(self, device: str = "cuda"):
        self.config: Dict = dict()
        self.CONFIG_FILE = "config.yaml"

        cwd = Path.cwd()
        root_dir = cwd.parent.parent.parent

        with open(path.join(root_dir, self.CONFIG_FILE), "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.DATA_DIR: str = path.join(
            root_dir, self.config[ConfigSettings.DATA_DIRECTORY]
        )

        self.CHECKPOINT_PATH: path = path.join(
            self.DATA_DIR,
            self.config[ConfigSettings.SQUAD_AGENT][ConfigSettings.CHECKPOINT_NAME],
        )

        ppo_settings: dict = self.config[ConfigSettings.SQUAD_AGENT][ConfigSettings.PPO]
        self.clip_coefficient: float = ppo_settings[ConfigSettings.CLIP_COEFFICIENT]
        self.entropy_coefficient: float = ppo_settings[
            ConfigSettings.ENTROPY_COEFFICIENT
        ]
        self.batch_size: int = ppo_settings[ConfigSettings.BATCH_SIZE]
        self.gae_lambda: float = ppo_settings[ConfigSettings.GAE_LAMBDA]
        self.gamma: float = ppo_settings[ConfigSettings.GAMMA]
        self.max_grad_norm: float = ppo_settings[ConfigSettings.MAX_GRAD_NORM]
        self.num_rollout_steps: int = ppo_settings[ConfigSettings.NUM_ROLLOUT_STEPS]
        self.update_policy_epochs: int = ppo_settings[
            ConfigSettings.UPDATE_POLICY_EPOCHS
        ]
        self.vf_coefficient: float = ppo_settings[ConfigSettings.VF_COEFFICIENT]

        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = ActorCritic(len(SQUAD_ACTIONS), self.device, None, 0, 0).to(
            self.device
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2.5e-4, eps=1e-5)

        if path.isfile(self.CHECKPOINT_PATH):
            self.model, self.optimizer, self.epoch = load_checkpoint(
                self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
            )
            logger.info(f"Loaded existing model at {self.CHECKPOINT_PATH}")

        self.model.train()

        self.initial_lstm_state = (
            torch.zeros(self.model.lstm.num_layers, 1, self.model.lstm.hidden_size).to(
                self.device
            ),
            torch.zeros(self.model.lstm.num_layers, 1, self.model.lstm.hidden_size).to(
                self.device
            ),
        )

        self.current_lstm_state = self.initial_lstm_state
        self.writer = SummaryWriter(path.join(self.DATA_DIR, "runs"))

        self.states_dir = path.join(
            self.DATA_DIR,
            self.config[ConfigSettings.SQUAD_AGENT][ConfigSettings.STATE_DIRECTORY],
        )

        if not path.exists(self.states_dir):
            makedirs(self.states_dir)

        self.remove_paths: List[path] = []

    def learn(self) -> None:
        for game_folder in listdir(self.states_dir):
            game_dir: path = path.join(self.states_dir, game_folder)
            self.remove_paths.append(game_dir)
            for tensors_file in listdir(game_dir):
                if tensors_file.endswith(".pt"):
                    tensors = torch.load(
                        path.join(self.states_dir, game_folder, tensors_file),
                        map_location=self.device,
                    )
                    self._back_propagation(tensors)

    def _back_propagation(self, tensors: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            actions = tensors["actions"]
            # actions = torch.autograd.Variable(actions, requires_grad=True)
            logprobs = tensors["logprobs"]
            # logprobs = torch.autograd.Variable(logprobs, requires_grad=True)
            spatials = tensors["spatials"]
            # spatials = torch.autograd.Variable(spatials, requires_grad=True)
            entities = tensors["entities"]
            # entities = torch.autograd.Variable(entities, requires_grad=True)
            scalars = tensors["scalars"]
            # scalars = torch.autograd.Variable(scalars, requires_grad=True)
            locations = tensors["locations"]
            # locations = torch.autograd.Variable(locations, requires_grad=True)
            dones = tensors["dones"]
            # dones = torch.autograd.Variable(dones, requires_grad=True)
            rewards = tensors["rewards"]
            # rewards = torch.autograd.Variable(rewards, requires_grad=True)
            values = tensors["values"]
            # values = torch.autograd.Variable(values, requires_grad=True)

            next_value = self.model.get_value(
                spatials[-1],
                entities[-1],
                scalars[-1],
                locations[-1],
                self.current_lstm_state,
                dones[-1],
                False,
            ).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(self.device)
            last_gaelam = 0
            rollout_steps: int = self.num_rollout_steps
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_non_terminal = 1.0 - 0
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - 0
                    next_values = values[t + 1]
                delta = (
                    rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - values[t]
                )
                advantages[t] = last_gaelam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gaelam
                )
            returns = advantages + values

            # flatten the batch
            b_entities = entities.reshape((-1,) + ENTITY_SHAPE)
            b_entities = torch.squeeze(b_entities)
            b_scalars = scalars.reshape((-1,) + SCALAR_SHAPE)
            b_scalars = torch.squeeze(b_scalars)
            b_spatials = spatials.reshape((-1,) + SPATIAL_SHAPE)
            b_spatials = torch.squeeze(b_spatials)
            b_locations = locations.reshape((-1,) + (256, 2))
            b_locations = torch.squeeze(b_locations)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_dones = dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # train the network
            # envsperbatch = NUM_ENVS // NUM_MINIBATCHES
            envinds = np.arange(1)
            flatinds = torch.arange(self.batch_size).reshape(rollout_steps, 1)
            # split into 4 minibatches
            mini_batch_ids = torch.chunk(flatinds, 4, dim=0)
            current_minibatch: int = 0
            for epoch in range(self.update_policy_epochs):
                # np.random.shuffle(envinds)
                if current_minibatch >= 4:
                    current_minibatch = 0
                mbenvinds = envinds
                # mb_inds = flatinds[:, mbenvinds].ravel()
                mb_inds = mini_batch_ids[current_minibatch].squeeze()
                current_minibatch += 1
                print(mb_inds)

                # mb_inds = np.random.randint(
                #     NUM_ROLLOUT_STEPS, size=(NUM_ROLLOUT_STEPS // 4,)
                # )

                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    _,
                    _,
                ) = self.model.get_action_and_value(
                    b_spatials[mb_inds],
                    b_entities[mb_inds],
                    b_scalars[mb_inds],
                    b_locations[mb_inds],
                    (
                        self.current_lstm_state[0][:, mbenvinds],
                        self.current_lstm_state[1][:, mbenvinds],
                    ),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    process_spatial=False,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                # normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                clip: float = self.clip_coefficient
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # clip vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip,
                    clip,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.entropy_coefficient * entropy_loss
                    + v_loss * self.vf_coefficient
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.requires_grad = True
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.optimizer.zero_grad()

                # if args.target_kl is not None:
                if approx_kl > 0.01:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.writer.add_scalar("rewards/mean_reward", rewards.mean(), self.epoch)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.epoch)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.epoch)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.epoch)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.epoch)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.epoch)
        self.epoch += 1


if __name__ == "__main__":
    learner = PPOTrainer()
    learner.learn()
