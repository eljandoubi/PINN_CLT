"""Hand-made faults for reference.py: every one must make the tests fail.

    uv run python scripts/mutate_reference.py

Each fault is one exact text replacement in a throw-away copy of the repository,
so the working tree is never touched. A fault counts as killed when pytest
exits with code 1 (a test failed) or runs out of time.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 300
TESTS = ["tests/test_reference.py", "tests/test_compare.py"]

# (name, text in reference.py, replacement)
FAULTS = [
    (
        "d66 factor 4 -> 2",
        "4.0 * d66 * np.kron(gram(bx, length, 1, 1)",
        "2.0 * d66 * np.kron(gram(bx, length, 1, 1)",
    ),
    (
        "d12 term 0 2 / 2 0 dropped",
        "+ np.kron(gram(bx, length, 0, 2), gram(by, width, 2, 0))",
        "",
    ),
    ("gram scale exponent", "scale ** (1 - a - b)", "scale ** (1 - a)"),
    ("load without length", "length * (bx[0] @ weights)", "(bx[0] @ weights)"),
    ("load without width", "width * (by[0] @ weights)", "(by[0] @ weights)"),
    ("phi'' lacks cross term", "2.0 * g1 * p1 + g * p2", "g * p2"),
    (
        "phi' lacks g1 term",
        "basis[1, i] = g1 * p0 + g * p1",
        "basis[1, i] = g * p1",
    ),
    ("g'' wrong", "g2 = 2.0 - 6.0 * s", "g2 = 2.0 - 4.0 * s"),
    (
        "legendre derivative factor",
        "return 2.0**order * legendre.legval",
        "return legendre.legval",
    ),
    (
        "free basis second derivative skipped",
        "for order in range(MAX_DERIVATIVE + 1):\n            basis[order, j]",
        "for order in range(2):\n            basis[order, j]",
    ),
    ("energy factor", "-0.5 * load @ solution", "-1.0 * load @ solution"),
    (
        "x derivative not scaled by length",
        "phi = clamped_basis(count_x, s)[dx] / self.length**dx",
        "phi = clamped_basis(count_x, s)[dx]",
    ),
    (
        "y derivative not scaled by width",
        "psi = free_basis(count_y, t)[dy] / self.width**dy",
        "psi = free_basis(count_y, t)[dy]",
    ),
    (
        "jacobi scaling of the load dropped",
        "np.linalg.solve(scaled, scale * load)",
        "np.linalg.solve(scaled, load)",
    ),
    (
        "too few quadrature nodes",
        "max(count_x, count_y) + 6",
        "max(count_x, count_y) - 2",
    ),
    (
        "fields returns w_y as w_x",
        '"w_x": self.evaluate(x, y, dx=1)',
        '"w_x": self.evaluate(x, y, dy=1)',
    ),
    (
        "moment scale uses d22",
        "moment_scale = d11 * w_scale / length**2",
        "moment_scale = d22 * w_scale / length**2",
    ),
    (
        "supported moment drops d12",
        'd11 * supported["w_xx"] + d12 * supported["w_yy"]',
        'd11 * supported["w_xx"]',
    ),
    (
        "free moment swaps d12 and d22",
        'd12 * edge["w_xx"] + d22 * edge["w_yy"]',
        'd22 * edge["w_xx"] + d12 * edge["w_yy"]',
    ),
    (
        "clamped slope not scaled by length",
        'clamped["w_x"] * length',
        'clamped["w_x"]',
    ),
]


def run_tests(workdir: Path) -> int:
    command = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"]
    try:
        run = subprocess.run(
            command + TESTS,
            cwd=workdir,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 1
    return run.returncode


def main() -> int:
    ignore = shutil.ignore_patterns(
        ".venv", ".git", "runs", "wandb", "results", "__pycache__", ".*_cache"
    )
    killed = 0
    with tempfile.TemporaryDirectory() as scratch:
        workdir = Path(scratch) / "repo"
        shutil.copytree(ROOT, workdir, ignore=ignore)
        target = workdir / "reference.py"
        original = target.read_text()
        for name, old, new in FAULTS:
            if original.count(old) != 1:
                raise SystemExit(f"anchor missing or repeated: {name}")
            mutant = original.replace(old, new)
            compile(mutant, "reference.py", "exec")
            target.write_text(mutant)
            code = run_tests(workdir)
            killed += code == 1
            print(f"{'killed  ' if code == 1 else 'SURVIVED'} {name} (exit {code})")
            target.write_text(original)
    print(f"{killed} of {len(FAULTS)} faults killed")
    return 0 if killed == len(FAULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
