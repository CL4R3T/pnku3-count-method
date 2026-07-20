"""
Puzzle Solver: Start from 1, reach target m.

=== COMPLETE RULE TABLE ===

Button values: A starts at 1, B starts at 2.
E stores a snapshot of A or B for later use.

 # | Operation      | Effect       | A-cost | Constraints
---+----------------+--------------+--------+---------------------------
 1 | A -> N         | N += a       |   1    | at most n times
 2 | B -> N         | N *= b       |   0    | unlimited
 3 | A -> B  (C)    | b = 3        |   1    | a=1, b=2, only ONCE total
 4 | B -> A  (D)    | a = a * b    |   0    | a=1 (self-limiting, 1 use)
 5 | E -> A         | save current a|  1    | unlimited
 6 | E -> B         | save current b|  0    | unlimited
 7 | savedA -> N    | N += val     |   1    | consumes stored value
 8 | savedA -> B    | b = 3        |   1    | stored val=1, b=2, once total
 9 | savedB -> N    | N *= val     |   0    | consumes stored value
10 | savedB -> A    | a = a * val  |   0    | a=1, consumes stored value

Key points:
  - B upgrades from 2->3 exactly once total (C or savedA->B).
  - B->A and savedB->A both require a=1; after one use a>=2 (self-limiting).
  - E can snapshot a/b unlimited times, enabling stored operations later.

Usage: python solve.py [n] [m] [-e]
       -e: enable operation E
"""

import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass(frozen=True)
class State:
    value: int
    a_left: int
    c_used: bool        # B has been upgraded 2->3 (once total)
    b_mult: int         # 2 or 3
    a_add: int          # 1, 2, 3, ...
    stored_a: Tuple[int, ...] = ()   # sorted
    stored_b: Tuple[int, ...] = ()   # sorted

    def __hash__(self):
        return hash((self.value, self.a_left, self.c_used,
                     self.b_mult, self.a_add, self.stored_a, self.stored_b))

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (self.value == other.value and self.a_left == other.a_left and
                self.c_used == other.c_used and self.b_mult == other.b_mult and
                self.a_add == other.a_add and
                self.stored_a == other.stored_a and
                self.stored_b == other.stored_b)

    def key(self):
        return (self.value, self.b_mult, self.a_add, self.c_used,
                self.stored_a, self.stored_b)


def _insert_sorted(t: Tuple[int, ...], v: int) -> Tuple[int, ...]:
    out = list(t)
    for i, x in enumerate(out):
        if v <= x:
            out.insert(i, v)
            return tuple(out)
    out.append(v)
    return tuple(out)


def _remove_one(t: Tuple[int, ...], v: int) -> Tuple[int, ...]:
    out = list(t)
    out.remove(v)
    return tuple(out)


def _try_transition(s: State, skey, best, path_of, queue,
                    new_val, desc, nkey_args):
    """Helper: if new state is better than best, add to queue."""
    nkey = nkey_args
    old = best.get(nkey, -1)
    a = s.a_left
    if desc.startswith('A(') or desc.startswith('C(') or desc.startswith('E(save a=') \
       or desc.startswith('SA(') or desc.startswith('SA->B('):
        a -= 1  # these cost 1 A
    if a > old:
        best[nkey] = a
        path_of[nkey] = path_of[skey] + (desc,)
        return True
    return False


def solve(target: int, n: int, enable_e: bool = False) -> Optional[List[str]]:
    """BFS to find a valid sequence of operations from 1 to target."""
    start = State(value=1, a_left=n, c_used=False, b_mult=2, a_add=1)

    best = {start.key(): n}
    path_of = {start.key(): ()}
    queue = deque([start])

    while queue:
        s = queue.popleft()
        skey = s.key()

        if s.value == target:
            return list(path_of[skey])

        if s.value > target:
            continue

        # ---- 1. A -> N ----
        if s.a_left > 0:
            new_val = s.value + s.a_add
            if new_val <= target:
                desc = f'A(+{s.a_add})'
                nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                        s.stored_a, s.stored_b)
                if s.a_left - 1 > best.get(nkey, -1):
                    best[nkey] = s.a_left - 1
                    path_of[nkey] = path_of[skey] + (desc,)
                    queue.append(State(value=new_val, a_left=s.a_left - 1,
                                      c_used=s.c_used, b_mult=s.b_mult,
                                      a_add=s.a_add,
                                      stored_a=s.stored_a,
                                      stored_b=s.stored_b))

        # ---- 2. B -> N ----
        new_val = s.value * s.b_mult
        if new_val <= target:
            desc = f'B(*{s.b_mult})'
            nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                    s.stored_a, s.stored_b)
            if s.a_left > best.get(nkey, -1):
                best[nkey] = s.a_left
                path_of[nkey] = path_of[skey] + (desc,)
                queue.append(State(value=new_val, a_left=s.a_left,
                                  c_used=s.c_used, b_mult=s.b_mult,
                                  a_add=s.a_add,
                                  stored_a=s.stored_a,
                                  stored_b=s.stored_b))

        # ---- 3. A -> B (C): b=3, cost 1A, a=1, b=2, once ----
        if s.a_left > 0 and not s.c_used and s.a_add == 1 and s.b_mult == 2:
            nkey = (s.value, 3, s.a_add, True, s.stored_a, s.stored_b)
            if s.a_left - 1 > best.get(nkey, -1):
                best[nkey] = s.a_left - 1
                path_of[nkey] = path_of[skey] + ('C(A->B: b=3)',)
                queue.append(State(value=s.value, a_left=s.a_left - 1,
                                  c_used=True, b_mult=3, a_add=s.a_add,
                                  stored_a=s.stored_a, stored_b=s.stored_b))

        # ---- 4. B -> A (D): a = a * b, free, a=1 ----
        if s.a_add == 1:
            new_a = s.a_add * s.b_mult
            nkey = (s.value, s.b_mult, new_a, s.c_used,
                    s.stored_a, s.stored_b)
            if s.a_left > best.get(nkey, -1):
                best[nkey] = s.a_left
                path_of[nkey] = path_of[skey] + (f'D(B->A: a={new_a})',)
                queue.append(State(value=s.value, a_left=s.a_left,
                                  c_used=s.c_used, b_mult=s.b_mult,
                                  a_add=new_a,
                                  stored_a=s.stored_a, stored_b=s.stored_b))

        if not enable_e:
            continue

        # ---- 5. E -> A: save current a_add, cost 1A ----
        if s.a_left > 0:
            new_sa = _insert_sorted(s.stored_a, s.a_add)
            nkey = (s.value, s.b_mult, s.a_add, s.c_used, new_sa, s.stored_b)
            if s.a_left - 1 > best.get(nkey, -1):
                best[nkey] = s.a_left - 1
                path_of[nkey] = path_of[skey] + (f'E(save a=+{s.a_add})',)
                queue.append(State(value=s.value, a_left=s.a_left - 1,
                                  c_used=s.c_used, b_mult=s.b_mult,
                                  a_add=s.a_add,
                                  stored_a=new_sa, stored_b=s.stored_b))

        # ---- 6. E -> B: save current b_mult, free ----
        new_sb = _insert_sorted(s.stored_b, s.b_mult)
        nkey = (s.value, s.b_mult, s.a_add, s.c_used, s.stored_a, new_sb)
        if s.a_left > best.get(nkey, -1):
            best[nkey] = s.a_left
            path_of[nkey] = path_of[skey] + (f'E(save b=*{s.b_mult})',)
            queue.append(State(value=s.value, a_left=s.a_left,
                              c_used=s.c_used, b_mult=s.b_mult,
                              a_add=s.a_add,
                              stored_a=s.stored_a, stored_b=new_sb))

        # ---- 7. savedA -> N: cost 1A, consume stored value ----
        if s.a_left > 0 and s.stored_a:
            seen = set()
            for sv in s.stored_a:
                if sv in seen:
                    continue
                seen.add(sv)
                new_val = s.value + sv
                if new_val <= target:
                    new_sa = _remove_one(s.stored_a, sv)
                    desc = f'SA(+{sv})'
                    nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                            new_sa, s.stored_b)
                    if s.a_left - 1 > best.get(nkey, -1):
                        best[nkey] = s.a_left - 1
                        path_of[nkey] = path_of[skey] + (desc,)
                        queue.append(State(value=new_val, a_left=s.a_left - 1,
                                          c_used=s.c_used, b_mult=s.b_mult,
                                          a_add=s.a_add,
                                          stored_a=new_sa, stored_b=s.stored_b))

        # ---- 8. savedA -> B: b=3, cost 1A, stored=1, b=2, once ----
        if s.a_left > 0 and not s.c_used and s.b_mult == 2 and s.stored_a:
            if 1 in s.stored_a:
                new_sa = _remove_one(s.stored_a, 1)
                desc = 'SA->B(b=3)'
                nkey = (s.value, 3, s.a_add, True, new_sa, s.stored_b)
                if s.a_left - 1 > best.get(nkey, -1):
                    best[nkey] = s.a_left - 1
                    path_of[nkey] = path_of[skey] + (desc,)
                    queue.append(State(value=s.value, a_left=s.a_left - 1,
                                      c_used=True, b_mult=3, a_add=s.a_add,
                                      stored_a=new_sa, stored_b=s.stored_b))

        # ---- 9. savedB -> N: free, consume stored value ----
        if s.stored_b:
            seen = set()
            for sv in s.stored_b:
                if sv in seen:
                    continue
                seen.add(sv)
                new_val = s.value * sv
                if new_val <= target:
                    new_sb = _remove_one(s.stored_b, sv)
                    desc = f'SB(*{sv})'
                    nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                            s.stored_a, new_sb)
                    if s.a_left > best.get(nkey, -1):
                        best[nkey] = s.a_left
                        path_of[nkey] = path_of[skey] + (desc,)
                        queue.append(State(value=new_val, a_left=s.a_left,
                                          c_used=s.c_used, b_mult=s.b_mult,
                                          a_add=s.a_add,
                                          stored_a=s.stored_a, stored_b=new_sb))

        # ---- 10. savedB -> A: a = a * val, free, a=1, consume stored ----
        if s.a_add == 1 and s.stored_b:
            seen = set()
            for sv in s.stored_b:
                if sv in seen:
                    continue
                seen.add(sv)
                new_a = s.a_add * sv   # = sv since a=1
                new_sb = _remove_one(s.stored_b, sv)
                desc = f'SB->A(a={new_a})'
                nkey = (s.value, s.b_mult, new_a, s.c_used,
                        s.stored_a, new_sb)
                if s.a_left > best.get(nkey, -1):
                    best[nkey] = s.a_left
                    path_of[nkey] = path_of[skey] + (desc,)
                    queue.append(State(value=s.value, a_left=s.a_left,
                                      c_used=s.c_used, b_mult=s.b_mult,
                                      a_add=new_a,
                                      stored_a=s.stored_a, stored_b=new_sb))

    return None


# ================================================================

def format_path(path: List[str], target: int) -> str:
    if not path:
        return "No solution found."

    lines = []
    val = 1
    a_count = 0
    lines.append(f"Start: 1")

    for i, step in enumerate(path):
        if step.startswith('A('):
            a_count += 1
            val += int(step.split('+')[1].rstrip(')'))
        elif step.startswith('B('):
            val *= int(step.split('*')[1].rstrip(')'))
        elif step.startswith('C('):
            a_count += 1
        elif step.startswith('E(save a='):
            a_count += 1
        elif step.startswith('SA('):
            a_count += 1
            if '->B' not in step:
                val += int(step.split('+')[1].rstrip(')'))
        elif step.startswith('SA->B('):
            a_count += 1
        elif step.startswith('SB('):
            val *= int(step.split('*')[1].rstrip(')'))
        lines.append(f"  {i+1:3d}. {step:22s} -> {val}")

    lines.append(f"\nResult: {val} (target: {target}) {'[OK]' if val == target else '[FAIL]'}")
    lines.append(f"A operations used: {a_count}")
    lines.append(f"Total steps: {len(path)}")
    return '\n'.join(lines)


def verify_path(path: List[str], target: int, n: int,
                enable_e: bool = False) -> Tuple[bool, int, int]:
    val = 1
    a_count = 0
    c_used = False
    b_mult = 2
    a_add = 1
    stored_a = []
    stored_b = []

    for step in path:
        if step.startswith('A('):
            a_count += 1
            val += a_add
        elif step.startswith('B('):
            val *= b_mult
        elif step.startswith('C('):
            a_count += 1
            if c_used or a_add != 1 or b_mult != 2:
                return (False, val, a_count)
            c_used = True
            b_mult = 3
        elif step.startswith('D('):
            if a_add != 1:
                return (False, val, a_count)
            a_add = a_add * b_mult
        elif step.startswith('E(save a='):
            a_count += 1
            if not enable_e:
                return (False, val, a_count)
            sv = int(step.split('+')[1].rstrip(')'))
            stored_a.append(sv)
        elif step.startswith('E(save b='):
            if not enable_e:
                return (False, val, a_count)
            sv = int(step.split('*')[1].rstrip(')'))
            stored_b.append(sv)
        elif step.startswith('SA->B('):
            a_count += 1
            if not enable_e or c_used or b_mult != 2:
                return (False, val, a_count)
            if 1 not in stored_a:
                return (False, val, a_count)
            stored_a.remove(1)
            c_used = True
            b_mult = 3
        elif step.startswith('SA('):
            a_count += 1
            sv = int(step.split('+')[1].rstrip(')'))
            if sv not in stored_a:
                return (False, val, a_count)
            stored_a.remove(sv)
            val += sv
        elif step.startswith('SB->A('):
            sv = int(step.split('(a=')[1].rstrip(')'))
            if sv not in stored_b or a_add != 1:
                return (False, val, a_count)
            stored_b.remove(sv)
            a_add = a_add * sv
        elif step.startswith('SB('):
            sv = int(step.split('*')[1].rstrip(')'))
            if sv not in stored_b:
                return (False, val, a_count)
            stored_b.remove(sv)
            val *= sv
        else:
            return (False, val, a_count)

    return (val == target and a_count <= n, val, a_count)


def find_all_solutions(target: int, n: int, enable_e: bool = False,
                       max_solutions: int = 10) -> List[List[str]]:
    """Same BFS as solve(), collects multiple solutions."""
    start = State(value=1, a_left=n, c_used=False, b_mult=2, a_add=1)
    best = {start.key(): n}
    path_of = {start.key(): ()}
    queue = deque([start])
    solutions = []

    while queue and len(solutions) < max_solutions:
        s = queue.popleft()
        skey = s.key()

        if s.value == target:
            path = list(path_of[skey])
            if path not in solutions:
                solutions.append(path)
            continue

        if s.value > target:
            continue

        # 1. A -> N
        if s.a_left > 0:
            new_val = s.value + s.a_add
            if new_val <= target:
                desc = f'A(+{s.a_add})'
                nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                        s.stored_a, s.stored_b)
                if s.a_left - 1 > best.get(nkey, -1):
                    best[nkey] = s.a_left - 1
                    path_of[nkey] = path_of[skey] + (desc,)
                    queue.append(State(value=new_val, a_left=s.a_left - 1,
                                      c_used=s.c_used, b_mult=s.b_mult,
                                      a_add=s.a_add,
                                      stored_a=s.stored_a,
                                      stored_b=s.stored_b))

        # 2. B -> N
        new_val = s.value * s.b_mult
        if new_val <= target:
            desc = f'B(*{s.b_mult})'
            nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                    s.stored_a, s.stored_b)
            if s.a_left > best.get(nkey, -1):
                best[nkey] = s.a_left
                path_of[nkey] = path_of[skey] + (desc,)
                queue.append(State(value=new_val, a_left=s.a_left,
                                  c_used=s.c_used, b_mult=s.b_mult,
                                  a_add=s.a_add,
                                  stored_a=s.stored_a,
                                  stored_b=s.stored_b))

        # 3. C: A -> B
        if s.a_left > 0 and not s.c_used and s.a_add == 1 and s.b_mult == 2:
            nkey = (s.value, 3, s.a_add, True, s.stored_a, s.stored_b)
            if s.a_left - 1 > best.get(nkey, -1):
                best[nkey] = s.a_left - 1
                path_of[nkey] = path_of[skey] + ('C(A->B: b=3)',)
                queue.append(State(value=s.value, a_left=s.a_left - 1,
                                  c_used=True, b_mult=3, a_add=s.a_add,
                                  stored_a=s.stored_a, stored_b=s.stored_b))

        # 4. D: B -> A
        if s.a_add == 1:
            new_a = s.a_add * s.b_mult
            nkey = (s.value, s.b_mult, new_a, s.c_used,
                    s.stored_a, s.stored_b)
            if s.a_left > best.get(nkey, -1):
                best[nkey] = s.a_left
                path_of[nkey] = path_of[skey] + (f'D(B->A: a={new_a})',)
                queue.append(State(value=s.value, a_left=s.a_left,
                                  c_used=s.c_used, b_mult=s.b_mult,
                                  a_add=new_a,
                                  stored_a=s.stored_a, stored_b=s.stored_b))

        if not enable_e:
            continue

        # 5. E -> A
        if s.a_left > 0:
            new_sa = _insert_sorted(s.stored_a, s.a_add)
            nkey = (s.value, s.b_mult, s.a_add, s.c_used, new_sa, s.stored_b)
            if s.a_left - 1 > best.get(nkey, -1):
                best[nkey] = s.a_left - 1
                path_of[nkey] = path_of[skey] + (f'E(save a=+{s.a_add})',)
                queue.append(State(value=s.value, a_left=s.a_left - 1,
                                  c_used=s.c_used, b_mult=s.b_mult,
                                  a_add=s.a_add,
                                  stored_a=new_sa, stored_b=s.stored_b))

        # 6. E -> B
        new_sb = _insert_sorted(s.stored_b, s.b_mult)
        nkey = (s.value, s.b_mult, s.a_add, s.c_used, s.stored_a, new_sb)
        if s.a_left > best.get(nkey, -1):
            best[nkey] = s.a_left
            path_of[nkey] = path_of[skey] + (f'E(save b=*{s.b_mult})',)
            queue.append(State(value=s.value, a_left=s.a_left,
                              c_used=s.c_used, b_mult=s.b_mult,
                              a_add=s.a_add,
                              stored_a=s.stored_a, stored_b=new_sb))

        # 7. savedA -> N
        if s.a_left > 0 and s.stored_a:
            seen = set()
            for sv in s.stored_a:
                if sv in seen:
                    continue
                seen.add(sv)
                new_val = s.value + sv
                if new_val <= target:
                    new_sa = _remove_one(s.stored_a, sv)
                    desc = f'SA(+{sv})'
                    nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                            new_sa, s.stored_b)
                    if s.a_left - 1 > best.get(nkey, -1):
                        best[nkey] = s.a_left - 1
                        path_of[nkey] = path_of[skey] + (desc,)
                        queue.append(State(value=new_val, a_left=s.a_left - 1,
                                          c_used=s.c_used, b_mult=s.b_mult,
                                          a_add=s.a_add,
                                          stored_a=new_sa, stored_b=s.stored_b))

        # 8. savedA -> B
        if s.a_left > 0 and not s.c_used and s.b_mult == 2 and s.stored_a:
            if 1 in s.stored_a:
                new_sa = _remove_one(s.stored_a, 1)
                desc = 'SA->B(b=3)'
                nkey = (s.value, 3, s.a_add, True, new_sa, s.stored_b)
                if s.a_left - 1 > best.get(nkey, -1):
                    best[nkey] = s.a_left - 1
                    path_of[nkey] = path_of[skey] + (desc,)
                    queue.append(State(value=s.value, a_left=s.a_left - 1,
                                      c_used=True, b_mult=3, a_add=s.a_add,
                                      stored_a=new_sa, stored_b=s.stored_b))

        # 9. savedB -> N
        if s.stored_b:
            seen = set()
            for sv in s.stored_b:
                if sv in seen:
                    continue
                seen.add(sv)
                new_val = s.value * sv
                if new_val <= target:
                    new_sb = _remove_one(s.stored_b, sv)
                    desc = f'SB(*{sv})'
                    nkey = (new_val, s.b_mult, s.a_add, s.c_used,
                            s.stored_a, new_sb)
                    if s.a_left > best.get(nkey, -1):
                        best[nkey] = s.a_left
                        path_of[nkey] = path_of[skey] + (desc,)
                        queue.append(State(value=new_val, a_left=s.a_left,
                                          c_used=s.c_used, b_mult=s.b_mult,
                                          a_add=s.a_add,
                                          stored_a=s.stored_a, stored_b=new_sb))

        # 10. savedB -> A
        if s.a_add == 1 and s.stored_b:
            seen = set()
            for sv in s.stored_b:
                if sv in seen:
                    continue
                seen.add(sv)
                new_a = s.a_add * sv
                new_sb = _remove_one(s.stored_b, sv)
                desc = f'SB->A(a={new_a})'
                nkey = (s.value, s.b_mult, new_a, s.c_used,
                        s.stored_a, new_sb)
                if s.a_left > best.get(nkey, -1):
                    best[nkey] = s.a_left
                    path_of[nkey] = path_of[skey] + (desc,)
                    queue.append(State(value=s.value, a_left=s.a_left,
                                      c_used=s.c_used, b_mult=s.b_mult,
                                      a_add=new_a,
                                      stored_a=s.stored_a, stored_b=new_sb))

    return solutions


# ================================================================
def main():
    enable_e = '-e' in sys.argv
    args = [a for a in sys.argv[1:] if a != '-e']

    if len(args) == 2:
        n = int(args[0])
        m = int(args[1])
        print(f"Solving for n={n}, m={m} (E={'ON' if enable_e else 'OFF'})")
        print("=" * 60)
        path = solve(m, n, enable_e)
        if path:
            print(format_path(path, m))
            ok, val, a_uses = verify_path(path, m, n, enable_e)
            print(f"Verification: {'PASS' if ok else 'FAIL'}")
        else:
            print("No solution found!")

        print()
        print("All solutions:")
        sols = find_all_solutions(m, n, enable_e, max_solutions=10)
        for i, sol in enumerate(sols):
            print(f"  {i+1}. {' -> '.join(sol)}")
        if not sols:
            print("  (none)")
    else:
        for label, n, m in [("Example 1", 1, 54), ("Example 2", 4, 766)]:
            print("=" * 70)
            print(f"  {label}: n={n}, m={m} (E={'ON' if enable_e else 'OFF'})")
            print("=" * 70)
            path = solve(m, n, enable_e)
            print(format_path(path, m))

            print()
            print("  All solutions:")
            sols = find_all_solutions(m, n, enable_e, max_solutions=10)
            for i, sol in enumerate(sols):
                ok, val, a = verify_path(sol, m, n, enable_e)
                print(f"    {i+1}. {' -> '.join(sol)}")
                print(f"       verify: {'OK' if ok else 'FAIL'}, A-uses: {a}/{n}")
            print()


if __name__ == '__main__':
    main()
