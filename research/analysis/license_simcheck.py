"""License audit: code-similarity check (2026-10-22, docs/LICENSE_AUDIT_2026-10-22.md): for every function / class body in the target package, find the share of its
token 6-grams that also appear in a reference corpus (e.g. ultralytics, D-FINE, DEIM).

Two views:
  exact  - tokens as written (identifiers kept)        -> catches verbatim / lightly edited copies
  shape  - identifiers and literals normalised          -> catches renamed copies / close translations

Usage: python -I simcheck.py TARGET_DIR REF_NAME=REF_DIR [REF_NAME=REF_DIR ...] [--min-tokens 40] [--thr 0.3]
"""
import ast, io, sys, tokenize, keyword, collections, os, json

N = 6


def iter_py(root):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".git", "tests", "test")]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)


def toks(src):
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                          tokenize.ENCODING, tokenize.ENDMARKER):
                continue
            if t.type == tokenize.STRING and len(t.string) > 40:      # docstrings / long strings
                continue
            out.append((t.type, t.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def exact(ts):
    return [s for _, s in ts]


def shape(ts):
    r = []
    for ty, s in ts:
        if ty == tokenize.NAME and not keyword.iskeyword(s):
            r.append("ID")
        elif ty == tokenize.NUMBER:
            r.append("NUM")
        elif ty == tokenize.STRING:
            r.append("STR")
        else:
            r.append(s)
    return r


def grams(seq):
    return {tuple(seq[i:i + N]) for i in range(len(seq) - N + 1)}


def units(path):
    """Yield (name, source) for each top-level function / method / class."""
    src = open(path, encoding="utf-8", errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return
    lines = src.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = "".join(lines[node.lineno - 1:node.end_lineno])
            yield f"{node.name}:{node.lineno}", seg


def build_ref(name, root):
    idx_e, idx_s = collections.defaultdict(set), collections.defaultdict(set)
    nfiles = 0
    for p in iter_py(root):
        nfiles += 1
        rel = os.path.relpath(p, root)
        for uname, seg in units(p):
            ts = toks(seg)
            key = f"{rel}::{uname}"
            for g in grams(exact(ts)):
                idx_e[g].add(key)
            for g in grams(shape(ts)):
                idx_s[g].add(key)
    print(f"[ref {name}] {nfiles} files, {len(idx_e)} exact grams, {len(idx_s)} shape grams", file=sys.stderr)
    return idx_e, idx_s


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    min_tokens = int(opts.get("--min-tokens", 40))
    thr = float(opts.get("--thr", 0.3))
    target = args[0]
    refs = {}
    for a in args[1:]:
        n, d = a.split("=", 1)
        refs[n] = build_ref(n, d)
    rows = []
    for p in iter_py(target):
        rel = os.path.relpath(p, target)
        for uname, seg in units(p):
            ts = toks(seg)
            if len(ts) < min_tokens:
                continue
            ge, gs = grams(exact(ts)), grams(shape(ts))
            if not ge:
                continue
            for rname, (ie, is_) in refs.items():
                hit_e = collections.Counter()
                for g in ge:
                    for k in ie.get(g, ()):
                        hit_e[k] += 1
                hit_s = collections.Counter()
                for g in gs:
                    for k in is_.get(g, ()):
                        hit_s[k] += 1
                be = hit_e.most_common(1)[0] if hit_e else ("", 0)
                bs = hit_s.most_common(1)[0] if hit_s else ("", 0)
                fe, fs = be[1] / len(ge), bs[1] / len(gs)
                if fe >= thr or fs >= max(thr, 0.5):
                    rows.append((rel, uname, len(ts), rname, round(fe, 2), be[0], round(fs, 2), bs[0]))
    rows.sort(key=lambda r: (-r[4], -r[6]))
    print(f"{'target file':40} {'unit':34} {'tok':>5} {'ref':8} {'exact':>5} {'shape':>5}  best match")
    for r in rows:
        print(f"{r[0]:40} {r[1]:34} {r[2]:5d} {r[3]:8} {r[4]:5.2f} {r[6]:5.2f}  {r[5]}  |  {r[7]}")
    print(f"\n{len(rows)} units above threshold (exact>={thr} or shape>={max(thr,0.5)})", file=sys.stderr)


if __name__ == "__main__":
    main()
