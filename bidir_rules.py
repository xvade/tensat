#!/usr/bin/env python3
"""Build a bidirectional egg rule set from the full un-curated TASO corpus.
Keep every forward LHS=>RHS. Add the reverse RHS=>LHS only when it is a valid
egg rewrite: every variable in the reversed RHS (= old LHS) must appear in the
reversed LHS (= old RHS). Dedup exact duplicates. This removes the direction bias
(the curation kept one direction per equivalence) without touching C++/Z3."""
import re, sys

def vars_of(s): return set(re.findall(r'\?input_\d+', s))

def main():
    inp, out = sys.argv[1], sys.argv[2]
    fwd, rev, seen = [], [], set()
    n_in = 0
    for line in open(inp):
        line = line.strip()
        if not line or '=>' not in line:
            continue
        n_in += 1
        lhs, rhs = line.split('=>', 1)
        if line not in seen:
            seen.add(line); fwd.append(line)
        # reverse: new LHS = rhs, new RHS = lhs; need vars(lhs) subset vars(rhs)
        # AND new-LHS must not be a bare variable (matches every e-class -> blowup;
        # egg rejects variable-only patterns anyway).
        rhs_bare = re.fullmatch(r'\(?\s*\?input_\d+\s*\)?', rhs.strip()) is not None
        if vars_of(lhs) <= vars_of(rhs) and not rhs_bare:
            r = f"{rhs}=>{lhs}"
            if r not in seen:
                seen.add(r); rev.append(r)
    with open(out, 'w') as f:
        f.write('\n'.join(fwd + rev))  # NO trailing newline: main.rs splits on
        # "\n" without filtering empties, so a trailing "" would panic "".parse()
    print(f"input rules: {n_in}")
    print(f"forward kept (deduped): {len(fwd)}")
    print(f"valid reverses added:   {len(rev)}")
    print(f"total bidirectional:    {len(fwd)+len(rev)}")

if __name__ == '__main__':
    main()
