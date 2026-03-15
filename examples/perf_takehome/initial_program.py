"""
Performance Take-Home: Optimize build_kernel for a VLIW SIMD processor.

The goal is to minimize CPU cycles for a parallel tree traversal algorithm.
The program runs on a simulated VLIW SIMD processor with these slot limits per cycle:
  - ALU: 12 scalar operations
  - VALU: 6 vector operations (8-element SIMD vectors)
  - Load: 2 memory reads
  - Store: 2 memory writes
  - Flow: 1 control flow instruction

Key optimization strategies:
  - VLIW parallelism: pack independent ops into single cycles
  - SIMD vectorization: process 8 batch elements per vector instruction
  - Loop unrolling to reduce overhead
  - Instruction scheduling to maximize ILP
  - Use multiply_add for certain hash stages
  - Minimize load/store by reusing scratch values

The algorithm:
  For each round (16 rounds):
    For each batch element (256 elements):
      1. Load idx and val from memory
      2. Load node_val = forest[idx]
      3. val = hash(val ^ node_val)  -- 6-stage hash
      4. idx = 2*idx + (1 if val%2==0 else 2)
      5. if idx >= n_nodes: idx = 0
      6. Store idx and val back

HASH_STAGES (each stage: op1, val1, op2, op3, val3):
  ("+", 0x7ED55D16, "+", "<<", 12)
  ("^", 0xC761C23C, "^", ">>", 19)
  ("+", 0x165667B1, "+", "<<", 5)
  ("+", 0xD3A2646C, "^", "<<", 9)
  ("+", 0xFD7046C5, "+", "<<", 3)
  ("^", 0xB55A4F09, "^", ">>", 16)

Each hash stage computes:
  tmp1 = op1(val, val1)
  tmp2 = op3(val, val3)
  val = op2(tmp1, tmp2)

Available instructions:
  ALU: (op, dst, src1, src2) where op in +, -, *, /, %, ^, &, |, <<, >>, ==, !=, <, >, <=, >=
       Also: ("multiply_add", dst, a, b, c) => dst = a*b + c
  VALU: same ops but on 8-element vectors, e.g. ("+", vdst, vsrc1, vsrc2)
  Load: ("load", dst_scratch, addr_scratch) - load mem[scratch[addr]] into scratch[dst]
        ("const", dst_scratch, value) - store constant into scratch[dst]
        ("vload", vdst, addr_scratch) - load 8 contiguous words from mem[scratch[addr]]
        ("vbroadcast", vdst, scalar_scratch) - broadcast scalar to all 8 vector lanes
  Store: ("store", addr_scratch, src_scratch) - store scratch[src] to mem[scratch[addr]]
         ("vstore", addr_scratch, vsrc) - store 8 words to mem[scratch[addr]]
  Flow:  ("select", dst, cond, true_val, false_val) - conditional select
         ("vselect", vdst, vcond, vtrue, vfalse) - vector conditional select
         ("jump", target_pc) - unconditional jump
         ("cond_jump", cond_scratch, target_pc) - conditional jump
         ("pause",) - pause execution (ignored in submission)
         ("halt",) - stop execution
  Debug: ("comment", text), ("compare", addr, key) - ignored in submission

Scratch space: 1536 words total. Each alloc_scratch(name, length) reserves contiguous words.
Vector scratch: alloc_scratch(name, 8) reserves 8 contiguous words for a vector register.

IMPORTANT VLIW NOTES:
  - All effects in a cycle are applied atomically at end of cycle
  - You can read a value and write to it in the same cycle
  - Multiple engines execute in parallel within one instruction bundle
  - An instruction bundle is a dict mapping engine names to lists of slots:
    e.g. {"alu": [(...), (...)], "load": [(...), (...)], "valu": [(...), (...)]}

Baseline (naive scalar): 147,734 cycles
Best known: 1,363 cycles (108x speedup)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "problem_src"))

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

from collections import defaultdict


# EVOLVE-BLOCK-START
class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})
# EVOLVE-BLOCK-END
