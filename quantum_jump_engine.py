import random
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import csv

class QuantumJumpEngine:
    def __init__(self, config=None, max_history=1000):
        if config:
            with open(config, 'r') as f:
                cfg = json.load(f)
                self.states = cfg["states"]
                self.energy = cfg["energy"]
                self.jump_map = {tuple(k): tuple(v) for k, v in cfg["jump_map"].items()}
        else:
            self.states = ["|ψ₀⟩", "|ψ₁⟩", "|ψ₂⟩", "|ψ₃⟩", "|ψ₄⟩", 
                           "|WH1_in⟩", "|WH1_a⟩", "|WH1_b⟩", "|WH1_out1⟩", "|WH1_out2⟩",
                           "|WH2_in⟩", "|WH2_a⟩", "|WH2_b⟩", "|WH2_out1⟩", "|WH2_out2⟩"]
            self.energy = {
                "|ψ₀⟩": 10, "|ψ₁⟩": 8, "|ψ₂⟩": 5, "|ψ₃⟩": 6, "|ψ₄⟩": 4,
                "|WH1_in⟩": 7, "|WH1_a⟩": 6, "|WH1_b⟩": 5, "|WH1_out1⟩": 3, "|WH1_out2⟩": 4,
                "|WH2_in⟩": 9, "|WH2_a⟩": 8, "|WH2_b⟩": 7, "|WH2_out1⟩": 2, "|WH2_out2⟩": 3
            }
            self.jump_map = {
                ("|ψ₀⟩", "|ψ₁⟩"): (1, 'init'), ("|ψ₁⟩", "|ψ₂⟩"): (1, 'x'),
                ("|ψ₁⟩", "|ψ₃⟩"): (2, 'y'), ("|ψ₁⟩", "|ψ₀⟩"): (5, 'return'),
                ("|ψ₂⟩", "|ψ₃⟩"): (2, 'side'), ("|ψ₂⟩", "|ψ₄⟩"): (3, 'z'),
                ("|ψ₃⟩", "|ψ₄⟩"): (2, 'z'), ("|ψ₃⟩", "|ψ₂⟩"): (2, 'loopback'),
                ("|ψ₃⟩", "|ψ₁⟩"): (2, 'cross'), ("|ψ₃⟩", "|ψ₀⟩"): (4, 'reset'),
                ("|ψ₄⟩", "|ψ₁⟩"): (2, 'loop'), ("|ψ₄⟩", "|ψ₀⟩"): (6, 'recycle'),
                ("|ψ₂⟩", "|WH1_in⟩"): (1, 'wormhole1'), 
                ("|WH1_in⟩", "|WH1_a⟩"): (0, 'wh1_step1'), ("|WH1_a⟩", "|WH1_b⟩"): (0, 'wh1_step2'),
                ("|WH1_b⟩", "|WH1_out1⟩"): (0, 'wh1_exit1'), ("|WH1_b⟩", "|WH1_out2⟩"): (0, 'wh1_exit2'),
                ("|ψ₃⟩", "|WH2_in⟩"): (1, 'wormhole2'),
                ("|WH2_in⟩", "|WH2_a⟩"): (0, 'wh2_step1'), ("|WH2_a⟩", "|WH2_b⟩"): (0, 'wh2_step2'),
                ("|WH2_b⟩", "|WH2_out1⟩"): (0, 'wh2_exit1'), ("|WH2_b⟩", "|WH2_out2⟩"): (0, 'wh2_exit2')
            }
        self.observe_every = 6
        self.alpha = 0.2
        self.omega = 0.1
        self.energy_limit = 25
        self.temperature = 1.0
        self.wormhole_prob = 0.5
        self.wormhole_energy_factor = 0.5
        self.wormhole_max_steps = 3
        self.wormhole_step_energy_cost = 1.0
        self.wormhole_safe_threshold = 2.0
        self.max_history = max_history
        self.max_event_coords = 1000  # 限制事件座標儲存量
        self.reset()

    def set_temperature(self, temperature):
        if temperature <= 0: raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def set_energy_limit(self, energy_limit):
        if energy_limit <= 0: raise ValueError("Energy limit must be positive.")
        self.energy_limit = energy_limit

    def set_alpha(self, alpha): self.alpha = alpha
    def set_omega(self, omega): self.omega = omega
    def set_observe_every(self, observe_every): self.observe_every = observe_every
    def set_wormhole_prob(self, prob): self.wormhole_prob = max(0.0, min(1.0, prob))
    def set_wormhole_energy_factor(self, factor): self.wormhole_energy_factor = max(0.0, factor)
    def set_wormhole_max_steps(self, steps): self.wormhole_max_steps = max(1, int(steps))
    def set_wormhole_step_energy_cost(self, cost): self.wormhole_step_energy_cost = max(0.0, cost)
    def set_wormhole_safe_threshold(self, threshold): self.wormhole_safe_threshold = max(0.0, threshold)

    def reset(self):
        self.current_state = "|ψ₀⟩"
        self.t = 0
        self.e = self.energy[self.current_state]
        self.step = 0
        self.history = deque(maxlen=self.max_history)
        self.flow = defaultdict(int)
        self.energy_history = deque([(0, self.e)], maxlen=self.max_history)
        self.wormhole_stats = {
            'WH1': {'count': 0, 'steps': [], 'collapses': 0, 'safe_exits': 0},
            'WH2': {'count': 0, 'steps': [], 'collapses': 0, 'safe_exits': 0},
            'energy_changes': []
        }
        self.event_coords = {
            'collapses': deque(maxlen=self.max_event_coords),
            'safe_exits': deque(maxlen=self.max_event_coords)
        }
        self.suggested_threshold = self.wormhole_step_energy_cost * 2  # 初始建議閾值

    def logic_jump(self):
        options = [k[1] for k in self.jump_map if k[0] == self.current_state]
        weights = [math.exp(-(self.energy[s] - self.energy[self.current_state]) / self.temperature) for s in options]
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        return random.choices(options, weights=norm_weights, k=1)[0]

    def observer_intervention(self, next_state):
        if self.step > 0 and self.step % self.observe_every == 0:
            return "|ψ₃⟩" if next_state == "|ψ₂⟩" else "|ψ₂⟩" if next_state == "|ψ₃⟩" else next_state
        return next_state

    def get_safest_exit(self, current, wh_id):
        exits = [k[1] for k in self.jump_map if k[0] == current and "out" in k[1]]
        if not exits:
            exits = [s for s in self.states if "out" in s and wh_id in s]
        return min(exits, key=lambda x: self.energy[x]) if exits else current

    def wormhole_jump(self, wh_id):
        initial_energy = self.e
        max_possible_steps = min(self.wormhole_max_steps, int(initial_energy / self.wormhole_step_energy_cost))
        wh_steps = random.randint(1, max_possible_steps) if max_possible_steps > 0 else 0
        current = self.current_state
        path = [current]
        total_delta_e = 0
        collapsed = False
        safe_exit = False

        for i in range(wh_steps):
            options = [k[1] for k in self.jump_map if k[0] == current]
            if not options:
                break
            next_state = random.choice(options)
            delta_e = self.energy[next_state] - self.energy[current]
            wormhole_energy_change = 0
            if "out" not in next_state:
                wormhole_energy_change = random.uniform(-self.wormhole_energy_factor, self.wormhole_energy_factor) * self.e
                delta_e += wormhole_energy_change - self.wormhole_step_energy_cost
                self.wormhole_stats['energy_changes'].append(wormhole_energy_change)

            if self.e + delta_e <= self.wormhole_step_energy_cost:
                next_state = self.get_safest_exit(current, wh_id)
                delta_e = self.energy[next_state] - self.energy[current]
                safe_exit = True
                self.wormhole_stats[wh_id]['safe_exits'] += 1
                break
            elif self.e <= self.wormhole_safe_threshold + self.wormhole_step_energy_cost:
                next_state = self.get_safest_exit(current, wh_id)
                delta_e = self.energy[next_state] - self.energy[current]
                safe_exit = True
                self.wormhole_stats[wh_id]['safe_exits'] += 1
                break

            self.e += delta_e
            total_delta_e += delta_e
            self.energy_history.append((self.step, self.e))
            self.history.append((self.step, self.t, current, next_state, 0, delta_e, f"wh{wh_id}_step"))
            current = next_state
            path.append(current)
            self.step += 1

        if collapsed or safe_exit:
            if not safe_exit:
                next_state = self.get_safest_exit(current, wh_id)
                delta_e = self.energy[next_state] - self.energy[current]
                self.wormhole_stats[wh_id]['collapses'] += 1
            self.e += delta_e
            total_delta_e += delta_e
            event = "collapse" if not safe_exit else "safe_exit"
            self.history.append((self.step, self.t, current, next_state, 0, delta_e, f"wh{wh_id}_{event}"))
            self.energy_history.append((self.step, self.e))
            self.event_coords['collapses' if not safe_exit else 'safe_exits'].append((self.step, self.e))
            self.step += 1
            current = next_state

        self.wormhole_stats[wh_id]['count'] += 1
        self.wormhole_stats[wh_id]['steps'].append(len(path) - 1)
        return current, total_delta_e

    def step_jump(self):
        if self.e >= self.energy_limit or self.e <= 0:
            return False
        next_state = self.observer_intervention(self.logic_jump())
        jump_key = (self.current_state, next_state)
        delta_t, direction = self.jump_map.get(jump_key, (1, 'undefined'))
        osc_factor = 1 + self.alpha * math.sin(self.omega * self.t)
        real_delta_t = max(1, round(delta_t * osc_factor))
        delta_e = self.energy[next_state] - self.energy[self.current_state]

        if "WH" in self.current_state and self.current_state.endswith("_in⟩") and random.random() < self.wormhole_prob:
            wh_id = self.current_state[1:4]
            next_state, wormhole_delta_e = self.wormhole_jump(wh_id)
            real_delta_t = 0
            delta_e = wormhole_delta_e
            jump_key = (self.current_state, next_state)

        self.t += real_delta_t
        self.e += delta_e
        self.step += 1
        self.flow[jump_key] += 1
        self.energy_history.append((self.step, self.e))
        self.history.append((self.step, self.t, self.current_state, next_state, real_delta_t, delta_e, direction))
        self.current_state = next_state
        return True

    def get_stats(self):
        state_visits = defaultdict(int)
        for _, _, state, _, _, _, _ in self.history:
            state_visits[state] += 1
        avg_energy_change = (sum(self.wormhole_stats['energy_changes']) / len(self.wormhole_stats['energy_changes'])) if self.wormhole_stats['energy_changes'] else 0
        self.suggested_threshold = self.wormhole_step_energy_cost * 2 + avg_energy_change
        stats = {
            "state_visits": dict(state_visits),
            "total_steps": self.step,
            "final_energy": self.e,
            "wormhole_usage": self.wormhole_stats,
            "suggested_threshold": self.suggested_threshold
        }
        return stats

    def export_history(self, filepath):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Step', 'Time', 'Current State', 'Next State', 'Δt', 'ΔEnergy', 'Direction'])
            writer.writerows(self.history)

    def export_stats(self, filepath):
        stats = self.get_stats()
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Steps', stats['total_steps']])
            writer.writerow(['Final Energy', stats['final_energy']])
            writer.writerow(['Suggested Safe Threshold', stats['suggested_threshold']])
            writer.writerow(['Wormhole WH1 Usage', stats['wormhole_usage']['WH1']['count']])
            writer.writerow(['Wormhole WH1 Avg Steps', 
                            sum(stats['wormhole_usage']['WH1']['steps']) / len(stats['wormhole_usage']['WH1']['steps']) 
                            if stats['wormhole_usage']['WH1']['steps'] else 0])
            writer.writerow(['Wormhole WH1 Collapses', stats['wormhole_usage']['WH1']['collapses']])
            writer.writerow(['Wormhole WH1 Safe Exits', stats['wormhole_usage']['WH1']['safe_exits']])
            writer.writerow(['Wormhole WH2 Usage', stats['wormhole_usage']['WH2']['count']])
            writer.writerow(['Wormhole WH2 Avg Steps', 
                            sum(stats['wormhole_usage']['WH2']['steps']) / len(stats['wormhole_usage']['WH2']['steps']) 
                            if stats['wormhole_usage']['WH2']['steps'] else 0])
            writer.writerow(['Wormhole WH2 Collapses', stats['wormhole_usage']['WH2']['collapses']])
            writer.writerow(['Wormhole WH2 Safe Exits', stats['wormhole_usage']['WH2']['safe_exits']])
            writer.writerow(['Average Wormhole Energy Change', 
                            sum(stats['wormhole_usage']['energy_changes']) / len(stats['wormhole_usage']['energy_changes']) 
                            if stats['wormhole_usage']['energy_changes'] else 0])
            writer.writerow(['State Visits', ''])
            for state, count in stats['state_visits'].items():
                writer.writerow([state, count])

class QuantumGUI:
    def __init__(self, engine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("Quantum Jump Engine v2.11")
        self.running = False
        self.visualization_enabled = True
        self.show_collapses = tk.BooleanVar(value=True)
        self.show_safe_exits = tk.BooleanVar(value=True)

        self.create_controls()
        self.setup_visualization()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def create_controls(self):
        notebook = ttk.Notebook(self.root)
        basic_frame = ttk.Frame(notebook)
        wormhole_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Params")
        notebook.add(wormhole_frame, text="Wormhole Params")
        notebook.grid(row=0, column=0, columnspan=4, sticky="ew")

        # Basic Params
        basic_controls = [
            ("Temperature:", 0.1, 5.0, 1.0, self.engine.set_temperature),
            ("Energy Limit:", 10, 50, 25, self.engine.set_energy_limit),
            ("Alpha:", 0.0, 1.0, 0.2, self.engine.set_alpha),
            ("Omega:", 0.0, 1.0, 0.1, self.engine.set_omega),
            ("Observe Every:", 1, 10, 6, self.engine.set_observe_every),
        ]
        for i, (label, min_val, max_val, init_val, setter) in enumerate(basic_controls):
            ttk.Label(basic_frame, text=label).grid(row=i, column=0)
            scale = ttk.Scale(basic_frame, from_=min_val, to=max_val, value=init_val, 
                              command=lambda v, s=setter: s(float(v)))
            scale.grid(row=i, column=1)

        # Wormhole Params
        wormhole_controls = [
            ("Wormhole Prob:", 0.0, 1.0, 0.5, self.engine.set_wormhole_prob),
            ("Wormhole Energy Factor:", 0.0, 2.0, 0.5, self.engine.set_wormhole_energy_factor),
            ("Wormhole Max Steps:", 1, 5, 3, self.engine.set_wormhole_max_steps),
            ("Wormhole Step Energy Cost:", 0.1, 5.0, 1.0, self.engine.set_wormhole_step_energy_cost),
        ]
        for i, (label, min_val, max_val, init_val, setter) in enumerate(wormhole_controls):
            ttk.Label(wormhole